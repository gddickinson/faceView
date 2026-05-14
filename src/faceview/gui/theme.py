"""Theme tokens + Fusion-palette dark/light/system switcher (U6).

Minimum-viable U6: a Tools → Theme menu that lets the user pick
Dark / Light / System. Choice is persisted via QSettings under
``faceview/theme``. The chip / pill colours in StatusPanel + the
chat-block headers keep their existing hex (they were chosen for
brand consistency, not theme) — this is a global Qt-widget palette
swap, not a rebuild of every coloured token in the app.

Per-widget token-driven theming is a follow-up that touches
StatusPanel pills, PerceptionPanel rows, chat blocks, and the
canvas in RoomMapPanel; not in this slice.
"""

from __future__ import annotations

from typing import Literal

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from faceview.core.logger import get_logger


log = get_logger("theme")


ThemeMode = Literal["dark", "light", "system"]


_SETTINGS_ORG = "faceview"
_SETTINGS_APP = "main"
_KEY = "theme/mode"


def _dark_palette() -> QPalette:
    pal = QPalette()
    bg = QColor("#181b22")
    bg_alt = QColor("#1e222b")
    text = QColor("#dde2ec")
    disabled = QColor("#7a8290")
    accent = QColor("#5e72e4")
    pal.setColor(QPalette.ColorRole.Window, bg)
    pal.setColor(QPalette.ColorRole.WindowText, text)
    pal.setColor(QPalette.ColorRole.Base, bg_alt)
    pal.setColor(QPalette.ColorRole.AlternateBase, bg)
    pal.setColor(QPalette.ColorRole.ToolTipBase, bg_alt)
    pal.setColor(QPalette.ColorRole.ToolTipText, text)
    pal.setColor(QPalette.ColorRole.Text, text)
    pal.setColor(QPalette.ColorRole.Button, bg_alt)
    pal.setColor(QPalette.ColorRole.ButtonText, text)
    pal.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Highlight, accent)
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link, QColor("#3a8eff"))
    # Disabled-state shadings.
    pal.setColor(QPalette.ColorGroup.Disabled,
                 QPalette.ColorRole.WindowText, disabled)
    pal.setColor(QPalette.ColorGroup.Disabled,
                 QPalette.ColorRole.Text, disabled)
    pal.setColor(QPalette.ColorGroup.Disabled,
                 QPalette.ColorRole.ButtonText, disabled)
    return pal


def _light_palette() -> QPalette:
    # An explicit light palette so "Light" overrides macOS Dark
    # globally. Close to Qt's default-light defaults.
    pal = QPalette()
    bg = QColor("#f0f0f0")
    bg_alt = QColor("#ffffff")
    text = QColor("#1a1d24")
    disabled = QColor("#9aa3b2")
    accent = QColor("#1a73e8")
    pal.setColor(QPalette.ColorRole.Window, bg)
    pal.setColor(QPalette.ColorRole.WindowText, text)
    pal.setColor(QPalette.ColorRole.Base, bg_alt)
    pal.setColor(QPalette.ColorRole.AlternateBase, bg)
    pal.setColor(QPalette.ColorRole.ToolTipBase, bg_alt)
    pal.setColor(QPalette.ColorRole.ToolTipText, text)
    pal.setColor(QPalette.ColorRole.Text, text)
    pal.setColor(QPalette.ColorRole.Button, bg_alt)
    pal.setColor(QPalette.ColorRole.ButtonText, text)
    pal.setColor(QPalette.ColorRole.BrightText, QColor("#000000"))
    pal.setColor(QPalette.ColorRole.Highlight, accent)
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link, accent)
    pal.setColor(QPalette.ColorGroup.Disabled,
                 QPalette.ColorRole.WindowText, disabled)
    pal.setColor(QPalette.ColorGroup.Disabled,
                 QPalette.ColorRole.Text, disabled)
    pal.setColor(QPalette.ColorGroup.Disabled,
                 QPalette.ColorRole.ButtonText, disabled)
    return pal


def apply_theme(mode: ThemeMode) -> None:
    """Apply the chosen palette to the live QApplication + persist."""
    app = QApplication.instance()
    if app is None:
        return
    # Fusion gives identical paint on every OS — saves us reasoning
    # about per-platform style quirks.
    try:
        app.setStyle("Fusion")
    except Exception:  # noqa: BLE001
        pass
    if mode == "dark":
        app.setPalette(_dark_palette())
    elif mode == "light":
        app.setPalette(_light_palette())
    else:  # "system"
        app.setPalette(QPalette())  # default — Qt picks per-OS
    QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(_KEY, mode)
    log.info("theme.applied", mode=mode)


def load_persisted() -> ThemeMode:
    raw = QSettings(_SETTINGS_ORG, _SETTINGS_APP).value(_KEY)
    if raw in ("dark", "light", "system"):
        return raw  # type: ignore[return-value]
    return "system"
