from __future__ import annotations

from muse2api.accounts import Account, AccountPool, AccountStatus, JsonAccountStore
from muse2api.accounts.keepalive import quota_exhausted, sweep
from muse2api.drivers.base import SessionInfo
from muse2api.drivers.mock import MockDriver
from muse2api.errors import UpstreamAuthError


class FakeDriver(MockDriver):
    """a0: healthy, a1: weekly limit used up, a2: dead cookies."""

    async def renew_session(self, account: Account) -> SessionInfo:
        if account.id == "a2":
            raise UpstreamAuthError("expired")
        return SessionInfo(ok=True, cookies={}, cookie_expires={}, meta={})

    async def quota(self, account: Account) -> dict:
        self.quota_calls = getattr(self, "quota_calls", []) + [account.id]
        used = 100 if account.id == "a1" else 10
        return {"weekly_used_percent": used, "additional_left": None}


async def test_sweep_renews_checks_quota_and_marks_accounts(tmp_path):
    pool = AccountPool(JsonAccountStore(tmp_path / "acc.json"))
    for i in range(3):
        await pool.upsert(Account(id=f"a{i}", cookies={"hatch_sess": "x"}))
    driver = FakeDriver(delay=0)

    await sweep(pool, driver, quota_cooldown=600)

    assert pool.get("a0").status == AccountStatus.ACTIVE
    assert pool.get("a1").status == AccountStatus.COOLING
    assert pool.get("a2").status == AccountStatus.INVALID
    assert driver.quota_calls == ["a0", "a1"]  # dead account is not queried
    assert pool.get("a0").meta["quota"]["weekly_used_percent"] == 10


def test_quota_exhausted():
    assert quota_exhausted({"weekly_used_percent": 100, "additional_left": None})
    assert quota_exhausted({"weekly_used_percent": 100, "additional_left": "0 tokens left"})
    assert not quota_exhausted({"weekly_used_percent": 100, "additional_left": "1B tokens left"})
    assert not quota_exhausted({"weekly_used_percent": 40})


async def test_sweep_rechecks_invalid_accounts_once_a_day(tmp_path):
    pool = AccountPool(JsonAccountStore(tmp_path / "acc.json"))
    # back: muse.ai reopened it; dead: still blocked (FakeDriver.quota raises for a2)
    await pool.upsert(Account(id="back", cookies={"hatch_sess": "x"},
                              status=AccountStatus.INVALID, last_error="muse.ai disconnected"))
    await pool.upsert(Account(id="a2", cookies={"hatch_sess": "x"},
                              status=AccountStatus.INVALID, last_error="muse.ai disconnected"))

    class Driver(FakeDriver):
        async def quota(self, account):
            if account.id == "a2":
                raise UpstreamAuthError("muse.ai disconnected")
            return await super().quota(account)

    driver = Driver(delay=0)
    await pool.upsert(Account(id="mine", cookies={"hatch_sess": "x"}, enabled=False,
                              status=AccountStatus.INVALID, last_error="muse.ai disconnected"))
    await sweep(pool, driver)
    assert pool.get("back").status == AccountStatus.ACTIVE
    assert pool.get("back").last_error is None
    assert pool.get("a2").status == AccountStatus.INVALID
    assert not pool.get("a2").enabled  # blocked -> switched off

    driver.quota_calls = []
    await sweep(pool, driver)  # a2 was checked moments ago: not again today
    assert driver.quota_calls == ["back"]

    # muse.ai reopens a2 the next day: it is rechecked although disabled, and comes back on
    pool.get("a2").meta["invalid_checked_at"] = 0
    Driver.quota = FakeDriver.quota
    driver.quota_calls = []
    await sweep(pool, driver)
    assert pool.get("a2").enabled and pool.get("a2").status == AccountStatus.ACTIVE
    assert "mine" not in driver.quota_calls  # switched off by the user: left alone
