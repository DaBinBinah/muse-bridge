"""In-dashboard browser login: admin routes + capture state machine.

The browser is faked (throwaway ChromiumProcess / CDP classes swapped in); the
manager, routes, polling loop and account creation all run for real. The
TestClient runs the app on its own loop/thread, so the capture task's progress
is observed by synchronously polling the shared session state.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from muse2api.accounts import login_capture
from muse2api.accounts.login_capture import LoginCaptureManager
from muse2api.accounts.pool import AccountPool
from muse2api.accounts.store import JsonAccountStore
from muse2api.config import Settings
from muse2api.drivers.browser.cdp import CDPError

FULL = [
    {"name": "hatch_sess", "value": "S1", "domain": ".muse.ai", "expires": 1893456000},
    {"name": "hatch_gw", "value": "G1", "domain": ".muse.ai", "expires": 1893456000},
    {"name": "hatch_vml", "value": "V1", "domain": ".muse.ai", "expires": 1893456000},
    {"name": "hatch_native_auth_device", "value": "N1", "domain": ".muse.ai",
     "expires": 1893456000},
    {"name": "other", "value": "x", "domain": ".example.com"},  # wrong domain, dropped
]


class FakeChromium:
    instances: list[FakeChromium] = []

    def __init__(self, exe, port, profile, headless=True, proxy=""):
        self.port = port
        self.profile = profile
        self.stopped = False
        FakeChromium.instances.append(self)

    async def start(self):
        return "ws://fake/browser"

    async def stop(self):
        self.stopped = True


class FakeSession:
    """Browser-level session: routes CDP calls against a shared cookie jar."""

    jar: list[dict] = []
    fail_get_cookies = False

    async def send(self, method, params=None, timeout=30.0):
        if method == "Target.createTarget":
            return {"targetId": "t1"}
        if method == "Network.getCookies":
            if FakeSession.fail_get_cookies:
                raise CDPError("cdp connection closed")
            return {"cookies": list(FakeSession.jar)}
        if method == "Network.enable":
            return {}
        raise AssertionError(f"unexpected CDP call {method}")

    async def close(self):
        pass


class FakeCDPSession:
    @classmethod
    async def connect(cls, ws_url):
        return FakeSession()


@pytest.fixture
def fake_browser(monkeypatch):
    FakeChromium.instances = []
    FakeSession.jar = []
    FakeSession.fail_get_cookies = False
    monkeypatch.setattr(login_capture, "ChromiumProcess", FakeChromium)
    monkeypatch.setattr(login_capture, "CDPSession", FakeCDPSession)
    monkeypatch.setattr(login_capture, "find_chromium", lambda configured="": "/fake/chrome")
    monkeypatch.setattr(login_capture, "POLL_INTERVAL", 0.02)
    return FakeSession


def _wait_status(svc, sid, want, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = svc.login.get(sid)
        if state and state.status == want:
            return state
        time.sleep(0.01)
    pytest.fail(f"login session never reached status {want!r}")


def test_login_capture_creates_account(client, admin, fake_browser):
    FakeSession.jar = list(FULL)
    r = client.post("/admin/accounts/login", headers=admin, json={"label": "via-browser"})
    assert r.status_code == 200
    sid = r.json()["id"]

    state = _wait_status(client.app.state.services, sid, "captured")
    assert state.missing == []
    acc = state.account
    assert acc["label"] == "via-browser"
    assert set(acc["cookies"]) >= {"hatch_sess", "hatch_gw", "hatch_vml",
                                   "hatch_native_auth_device"}
    assert "other" not in acc["cookies"]  # non-muse.ai cookie dropped
    # browser closed and the throwaway profile removed
    assert FakeChromium.instances[-1].stopped
    assert not FakeChromium.instances[-1].profile.exists()
    # the account really is in the pool
    ids = [a["id"] for a in client.get("/admin/accounts", headers=admin).json()["data"]]
    assert acc["id"] in ids


def test_login_capture_refreshes_same_account(client, admin, fake_browser):
    FakeSession.jar = list(FULL)
    r1 = client.post("/admin/accounts/login", headers=admin, json={"label": "one"})
    first = _wait_status(client.app.state.services, r1.json()["id"], "captured")

    # re-login with the same hatch_sess but a rotated hatch_gw
    FakeSession.jar = [dict(c, value="G2") if c["name"] == "hatch_gw" else c
                       for c in FULL]
    r2 = client.post("/admin/accounts/login", headers=admin, json={})
    second = _wait_status(client.app.state.services, r2.json()["id"], "captured")

    assert second.account["id"] == first.account["id"]
    assert second.created_account is False
    # public() masks values; the reveal endpoint shows the rotated cookie stuck
    plain = client.get(f"/admin/accounts/{second.account['id']}/cookies",
                       headers=admin).json()["cookies"]
    assert plain["hatch_gw"] == "G2"


def test_login_capture_partial_cookies(client, admin, fake_browser):
    """A fresh muse.ai login only issues hatch_sess + hatch_native_auth_device
    (gw/vml never appear) — that still captures and creates a working account."""
    FakeSession.jar = [c for c in FULL if c["name"] in ("hatch_sess", "hatch_native_auth_device")]
    r = client.post("/admin/accounts/login", headers=admin, json={"label": "fresh-login"})
    state = _wait_status(client.app.state.services, r.json()["id"], "captured")
    assert state.missing == []
    plain = client.get(f"/admin/accounts/{state.account['id']}/cookies",
                       headers=admin).json()["cookies"]
    assert plain["hatch_sess"] == "S1"
    assert plain["hatch_native_auth_device"] == "N1"


def test_login_capture_two_different_accounts(client, admin, fake_browser):
    """The multi-account workflow: consecutive logins with different sessions
    each get their own account in the pool."""
    FakeSession.jar = list(FULL)
    r1 = client.post("/admin/accounts/login", headers=admin, json={"label": "acc1"})
    first = _wait_status(client.app.state.services, r1.json()["id"], "captured")

    # a second muse.ai account: different hatch_sess (and rotated hatch_gw)
    FakeSession.jar = [dict(c, value={"hatch_sess": "S2", "hatch_gw": "G2",
                                      "hatch_vml": "V2", "hatch_native_auth_device": "N2"}
                            [c["name"]]) for c in FULL if c["name"] != "other"]
    r2 = client.post("/admin/accounts/login", headers=admin, json={"label": "acc2"})
    second = _wait_status(client.app.state.services, r2.json()["id"], "captured")

    assert second.account["id"] != first.account["id"]
    assert second.created_account is True
    assert second.account["label"] == "acc2"
    accounts = client.get("/admin/accounts", headers=admin).json()["data"]
    assert len(accounts) == 2


def test_login_capture_cancel(client, admin, fake_browser):
    FakeSession.jar = []  # nothing logs in
    r = client.post("/admin/accounts/login", headers=admin, json={})
    sid = r.json()["id"]
    d = client.delete(f"/admin/accounts/login/{sid}", headers=admin)
    assert d.status_code == 200
    assert d.json()["status"] == "cancelled"
    assert FakeChromium.instances[-1].stopped


def test_login_capture_reports_window_closed(client, admin, fake_browser):
    FakeSession.jar = [c for c in FULL if c["name"] != "hatch_native_auth_device"]
    FakeSession.fail_get_cookies = True  # the user closed the browser
    r = client.post("/admin/accounts/login", headers=admin, json={})
    state = _wait_status(client.app.state.services, r.json()["id"], "cancelled")
    assert "closed" in (state.error or "")


def test_login_capture_timeout(fake_browser, tmp_path):
    FakeSession.jar = []
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    pool = AccountPool(JsonAccountStore(settings.accounts_file))
    mgr = LoginCaptureManager(settings, pool)

    async def run() -> str:
        info = await mgr.start("", timeout=0.05)
        for _ in range(300):  # deadline ~3s, in poll-sized steps
            if mgr.get(info["id"]).status != "waiting":
                break
            await asyncio.sleep(0.01)
        return mgr.get(info["id"]).status

    assert asyncio.run(run()) == "timeout"


def test_login_capture_one_window_at_a_time(client, admin, fake_browser):
    FakeSession.jar = []
    r1 = client.post("/admin/accounts/login", headers=admin, json={})
    assert r1.status_code == 200
    r2 = client.post("/admin/accounts/login", headers=admin, json={})
    assert r2.status_code == 400
    assert "already open" in r2.json()["error"]["message"]
    client.delete(f"/admin/accounts/login/{r1.json()['id']}", headers=admin)


def test_login_capture_unknown_session(client, admin, fake_browser):
    assert client.get("/admin/accounts/login/nope", headers=admin).status_code == 404
    assert client.delete("/admin/accounts/login/nope", headers=admin).status_code == 404
