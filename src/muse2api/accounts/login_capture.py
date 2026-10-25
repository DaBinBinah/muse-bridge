"""In-dashboard muse.ai login: a visible throwaway Chromium window.

Fishing all four ``hatch_*`` cookies out of devtools is fiddly — ``hatch_gw``
is regularly reported missing on machines that were never logged in on that
device. The dashboard can instead pop a real browser window on the server; the
user signs in there, the cookies (including HttpOnly ones devtools hides from
``document.cookie``) are read over CDP, and the account is created or
refreshed automatically.

The login window runs with its own temporary profile and its own CDP port and
never touches the proxy's persistent browser profile.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..drivers.browser.cdp import CDPError, CDPSession
from ..drivers.browser.chromium import ChromiumProcess, find_chromium
from ..upstream.muse import SESSION_COOKIES, SITE
from .model import Account, AccountStatus

log = logging.getLogger(__name__)

POLL_INTERVAL = 2.0
DEFAULT_TIMEOUT = 300.0


class LoginCaptureError(RuntimeError):
    """A login window is already open, or the browser could not be started."""


@dataclass
class CaptureState:
    id: str
    label: str
    status: str = "waiting"  # waiting | captured | timeout | cancelled | error
    missing: list[str] = field(default_factory=lambda: list(SESSION_COOKIES))
    error: str | None = None
    account: dict | None = None
    """``Account.public()`` once the cookies have been saved."""
    created_account: bool = True
    """False when an existing account with the same hatch_sess was refreshed."""
    created_at: float = field(default_factory=time.time)
    cookies: dict[str, str] = field(default_factory=dict)
    """Last cookies seen; only names are ever exposed via ``info()``."""
    expires: dict[str, int] = field(default_factory=dict)
    """Expiry timestamps of the last cookies seen."""

    def info(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "status": self.status,
                "missing": self.missing, "error": self.error,
                "account": self.account, "created_account": self.created_account}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _muse_cookies(all_cookies: list[dict]) -> tuple[dict[str, str], dict[str, int]]:
    """Filter a CDP cookie dump down to muse.ai session cookies (+ expiry)."""
    cookies: dict[str, str] = {}
    expires: dict[str, int] = {}
    for c in all_cookies:
        name, value = c.get("name", ""), c.get("value", "")
        domain = (c.get("domain") or "").lstrip(".")
        if not name or not value or not domain.endswith("muse.ai"):
            continue
        cookies[name] = value
        if exp := c.get("expires"):
            try:
                expires[name] = int(exp)
            except (TypeError, ValueError):
                pass
    return cookies, expires


class LoginCaptureManager:
    """One login window at a time; the dashboard polls the session info."""

    def __init__(self, settings: Settings, pool: Any) -> None:
        self._settings = settings
        self._pool = pool
        self._state: CaptureState | None = None
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    @property
    def current(self) -> CaptureState | None:
        return self._state

    async def start(self, label: str = "", timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
        async with self._lock:
            if self._state and self._state.status == "waiting":
                raise LoginCaptureError(
                    "a login window is already open — finish or cancel it first")
            state = CaptureState(id="login_" + uuid.uuid4().hex[:10], label=label)
            self._state = state
            self._task = asyncio.create_task(self._run(state, timeout))
            return state.info()

    def get(self, session_id: str) -> CaptureState | None:
        state = self._state
        return state if state and state.id == session_id else None

    async def cancel(self, session_id: str) -> CaptureState:
        state = self.get(session_id)
        if state is None:
            raise KeyError(session_id)
        if state.status == "waiting":
            state.status = "cancelled"
            if self._task:
                await asyncio.gather(self._task, return_exceptions=True)
        return state

    # ------------------------------------------------------------- internals

    async def _run(self, state: CaptureState, timeout: float) -> None:
        profile = self._settings.data_dir / "login-profiles" / state.id
        chromium: ChromiumProcess | None = None
        page: CDPSession | None = None
        browser: CDPSession | None = None
        try:
            chromium = ChromiumProcess(
                find_chromium(self._settings.chromium_path), _free_port(), profile,
                headless=False, proxy=self._settings.browser_proxy,
            )
            ws_url = await chromium.start()
            browser = await CDPSession.connect(ws_url)
            target = await browser.send("Target.createTarget", {"url": SITE})
            page = await CDPSession.connect(
                f"ws://127.0.0.1:{chromium.port}/devtools/page/{target['targetId']}"
            )
            await page.send("Network.enable")
            deadline = time.time() + timeout
            while state.status == "waiting" and time.time() < deadline:
                await asyncio.sleep(POLL_INTERVAL)
                try:
                    # Page-level getCookies (browser-level Network.getAllCookies
                    # was removed in Chrome 156) includes HttpOnly cookies and
                    # survives the login page navigating; auth popups land in
                    # the same browser context, so their cookies show up too.
                    res = await page.send("Network.getCookies", {"urls": [SITE]})
                    cookies, expires = _muse_cookies(res.get("cookies", []))
                except CDPError:
                    # The tab/window was closed. Complete from what we last saw.
                    if not state.missing:
                        state.status = "captured"
                    else:
                        state.status = "cancelled"
                        state.error = "browser window was closed before login completed"
                    break
                state.cookies = cookies
                state.expires = expires
                state.missing = [n for n in SESSION_COOKIES if not cookies.get(n)]
                if not state.missing:
                    state.status = "captured"
                    break
            if state.status == "captured":
                state.account, state.created_account = await self._save(
                    state.label, state.cookies, state.expires)
            elif state.status == "waiting":
                state.status = "timeout"
                state.error = f"no login within {int(timeout)} seconds"
        except FileNotFoundError as exc:
            state.status, state.error = "error", str(exc)
        except (RuntimeError, OSError) as exc:
            state.status = "error"
            state.error = (
                f"could not open a visible browser window ({exc}); "
                "the server may have no GUI — use cookie paste instead")
        except Exception as exc:  # noqa: BLE001
            log.exception("login capture failed")
            state.status, state.error = "error", str(exc)
        finally:
            for closer in (page, browser):
                if closer:
                    try:
                        await closer.close()
                    except Exception:  # noqa: BLE001
                        pass
            if chromium:
                try:
                    await chromium.stop()
                except Exception:  # noqa: BLE001
                    pass
            shutil.rmtree(profile, ignore_errors=True)

    async def _save(self, label: str, cookies: dict[str, str],
                    expires: dict[str, int]) -> tuple[dict, bool]:
        """Create an account, or refresh an existing one with the same session."""
        for acc in self._pool.all():
            if (acc.cookies.get("hatch_sess") and acc.cookies["hatch_sess"] == cookies.get("hatch_sess")):
                acc.cookies = {**acc.cookies, **cookies}
                acc.cookie_expires = {**acc.cookie_expires, **expires}
                # Fresh cookies give an invalid account another chance.
                acc.status, acc.last_error = AccountStatus.ACTIVE, None
                await self._pool.upsert(acc)
                return acc.public(), False
        acc = Account(label=label or time.strftime("login %m-%d %H:%M"),
                      cookies=cookies, cookie_expires=expires)
        await self._pool.upsert(acc)
        return acc.public(), True
