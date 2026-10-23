from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import Settings
from .base import MuseDriver

if TYPE_CHECKING:
    from ..core.media import MediaStore


def create_driver(settings: Settings, media: MediaStore | None = None) -> MuseDriver:
    """Instantiate the driver selected by ``MUSE2API_DRIVER``. Imports are lazy so that
    optional heavy dependencies of one driver never affect the others."""
    if settings.driver == "mock":
        from .mock import MockDriver

        return MockDriver()
    if settings.driver == "browser":
        from .browser.driver import BrowserDriver

        return BrowserDriver(settings, media=media)
    if settings.driver == "http":
        from .http.driver import HttpDriver

        return HttpDriver(settings)
    raise ValueError(f"unknown driver: {settings.driver}")
