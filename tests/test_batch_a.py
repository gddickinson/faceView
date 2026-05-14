"""S7 + C10 + U5 — API token auth, chat export, shortcuts dialog."""

from __future__ import annotations

from types import SimpleNamespace


# ── S7 — token auth middleware ──────────────────────────────────


def _build_test_api(monkeypatch, token: str | None = None):
    """Spin a FastAPI app pointed at a stub service so we can exercise
    the middleware without spinning up the GUI."""
    from fastapi.testclient import TestClient
    from faceview.server.api import build_app

    if token is None:
        monkeypatch.delenv("FACEVIEW_API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("FACEVIEW_API_TOKEN", token)

    class _StubService:
        def __init__(self):
            self.window = SimpleNamespace(llm_client=None)
        def get_camera_state(self): return {}
        def list_events(self, n=50): return []
        def list_chat_log(self, n=50): return []
        def monitor_snapshot(self, **kw): return {}
        def get_memory(self, **kw): return {}
        def clear_memory(self): return {}
        def set_engine(self, *a, **kw): return {}
        def set_test_engine(self, *a, **kw): return {}
        def set_lifecycle(self, *a, **kw): return {}
        def shutdown(self): return {}
        def send_chat(self, *a, **kw): return {}
        def speak(self, *a, **kw): return {}
        def screenshot(self, *a, **kw): return {}
        def set_emotion(self, *a, **kw): return {}
        def set_persona(self, *a, **kw): return {}
        def avatar_say(self, *a, **kw): return {}
        def list_personas(self): return []
        def list_effects(self): return []
        def list_active_effects(self): return []
        def trigger_effect(self, *a, **kw): return {}
        def stop_effect(self, *a, **kw): return {}
        def stop_all_effects(self): return {}
        def get_sliders(self): return {}
        def set_slider(self, *a, **kw): return {}
        def export_chat(self, **kw):
            import time as _time
            return {"ok": True, "exported_at": _time.time(),
                    "chat": [], "memory": {},
                    "persona": "test", "engine": "demo"}

    app = build_app(_StubService())
    return TestClient(app)


def test_auth_disabled_when_no_token(monkeypatch):
    api = _build_test_api(monkeypatch, token=None)
    # /chat/log requires no auth.
    resp = api.get("/chat/log")
    assert resp.status_code == 200


def test_auth_required_when_token_set(monkeypatch):
    api = _build_test_api(monkeypatch, token="s3cret")
    # Without header → 401.
    assert api.get("/chat/log").status_code == 401
    # With wrong header → 401.
    assert api.get(
        "/chat/log", headers={"x-api-token": "wrong"},
    ).status_code == 401
    # With right header → 200.
    assert api.get(
        "/chat/log", headers={"x-api-token": "s3cret"},
    ).status_code == 200


def test_auth_accepts_authorization_bearer(monkeypatch):
    api = _build_test_api(monkeypatch, token="s3cret")
    assert api.get(
        "/chat/log",
        headers={"authorization": "Bearer s3cret"},
    ).status_code == 200


def test_auth_bypass_for_healthz(monkeypatch):
    api = _build_test_api(monkeypatch, token="s3cret")
    # /healthz is open even with auth on — so SDKs can probe.
    resp = api.get("/healthz")
    assert resp.status_code == 200


# ── C10 — chat export ──────────────────────────────────────────


def test_chat_export_returns_chat_and_memory(monkeypatch):
    api = _build_test_api(monkeypatch, token=None)
    resp = api.get("/chat/export")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "chat" in body
    assert "memory" in body
    assert "exported_at" in body


# ── U5 — shortcuts dialog ──────────────────────────────────────


def test_shortcuts_dialog_renders_all_rows(qtbot):
    from PySide6.QtWidgets import QTableWidget
    from faceview.gui.shortcuts_dialog import ShortcutsDialog, SHORTCUTS
    dlg = ShortcutsDialog()
    qtbot.addWidget(dlg)
    tables = dlg.findChildren(QTableWidget)
    assert len(tables) == 1
    assert tables[0].rowCount() == len(SHORTCUTS)


def test_shortcuts_dialog_categories_present(qtbot):
    from faceview.gui.shortcuts_dialog import SHORTCUTS
    cats = {row[0] for row in SHORTCUTS}
    # Sanity: the major menu categories are covered.
    for cat in ("Chat", "Tools", "View", "File", "Help"):
        assert cat in cats


def test_main_window_help_shortcuts_action(qtbot):
    from faceview.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    # The action should exist + opening the dialog shouldn't raise.
    w._open_shortcuts_dialog()
    assert w._shortcuts_dialog is not None
    assert w._shortcuts_dialog.isVisible()
