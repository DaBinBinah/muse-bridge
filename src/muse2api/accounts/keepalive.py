"""Background session keepalive.

muse.ai sessions expire unless refreshed; this loop periodically asks the active
driver to renew each account and merges refreshed cookies back into the pool.
Drivers that do not support renewal simply raise ``FeatureNotImplemented`` and
the loop skips them.
"""

from __future__ import annotations

import asyncio
import logging
import time

from ..drivers.base import MuseDriver
from ..errors import FeatureNotImplemented, UpstreamAuthError
from .model import Account, AccountStatus
from .pool import AccountPool

log = logging.getLogger(__name__)


async def renew_account(pool: AccountPool, driver: MuseDriver, account_id: str) -> dict:
    acc = pool.get(account_id)
    if acc is None:
        return {"ok": False, "error": "not found"}
    try:
        info = await driver.renew_session(acc)
    except FeatureNotImplemented as exc:
        return {"ok": False, "error": exc.message}
    except UpstreamAuthError as exc:
        acc.status = AccountStatus.INVALID
        acc.last_error = exc.message
        await pool.upsert(acc)
        return {"ok": False, "error": exc.message}

    acc.cookies.update(info.cookies)
    acc.cookie_expires.update(info.cookie_expires)
    acc.meta.update(info.meta)
    acc.meta["renewed_at"] = time.time()
    await pool.upsert(acc)
    if info.ok:
        return {"ok": True, "meta": info.meta}
    # The request succeeded but the session was not assigned; surface why.
    return {"ok": False, "meta": info.meta,
            "error": f"session status: {info.meta.get('session_status', 'unknown')}"}


def quota_exhausted(snapshot: dict) -> bool:
    """Weekly limit used up and no additional tokens left to fall back on."""
    left = str(snapshot.get("additional_left") or "").strip()
    return (snapshot.get("weekly_used_percent") or 0) >= 100 and (not left or left.startswith("0"))


async def check_quota(pool: AccountPool, driver: MuseDriver, acc: Account,
                      cooldown: float = 3600) -> dict:
    """Read the account's quota, store it in ``meta`` and cool the account down
    when it is used up, so requests skip it instead of failing on it."""
    try:
        snapshot = await driver.quota(acc)
    except UpstreamAuthError as exc:
        acc.status, acc.last_error = AccountStatus.INVALID, exc.message
        await pool.upsert(acc)
        raise
    acc.meta["quota"] = {"checked_at": time.time(), **snapshot}
    if quota_exhausted(snapshot) and acc.status != AccountStatus.INVALID:
        acc.status = AccountStatus.COOLING
        acc.cooldown_until = max(acc.cooldown_until, time.time() + cooldown)
        log.warning("account %s is out of quota, cooling down", acc.id)
    await pool.upsert(acc)
    return acc.meta["quota"]


async def sweep(pool: AccountPool, driver: MuseDriver, quota_cooldown: float = 3600) -> None:
    """Renew every usable account's session, then check its quota."""
    for acc in pool.all():
        if not acc.enabled or acc.status == AccountStatus.INVALID:
            continue
        try:
            result = await renew_account(pool, driver, acc.id)
            log.info("keepalive %s -> %s", acc.id, result.get("ok"))
        except Exception:  # noqa: BLE001
            log.exception("keepalive failed for %s", acc.id)
        if acc.status == AccountStatus.INVALID or not driver.capabilities.quota:
            continue
        try:
            q = await check_quota(pool, driver, acc, quota_cooldown)
            log.info("quota %s -> weekly %s%%, extra %s", acc.id,
                     q.get("weekly_used_percent"), q.get("additional_left"))
        except Exception as exc:  # noqa: BLE001
            log.warning("quota check failed for %s: %s", acc.id, exc)


async def keepalive_loop(pool: AccountPool, driver: MuseDriver, interval: float) -> None:
    log.info("keepalive loop started (interval=%ss)", interval)
    while True:
        # First pass right at startup, so dead or exhausted accounts are known
        # before a user request lands on them. Cooldown lasts until the next pass.
        await sweep(pool, driver, quota_cooldown=interval)
        await asyncio.sleep(interval)
