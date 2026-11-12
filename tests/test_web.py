"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from faceview.web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# ── faceView-specific ─────────────────────────────────────────────────────

def _close(a, b, path="$"):
    """Recursive comparison with a float tolerance (libm differs in the last digit across platforms)."""
    if isinstance(b, float) or isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12), path
    elif isinstance(b, dict):
        assert set(a) == set(b), path
        for k in b:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(b, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    else:
        assert a == b, path


def test_reference_fixture_is_current():
    """tests/fixtures/face_reference.json matches what the Python code produces now (regenerate with
    scripts/make_face_fixture.py if the animation code changes)."""
    import json
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_face_fixture
    fresh = json.loads(json.dumps(make_face_fixture.make_reference()))
    _close(fresh, json.loads((ROOT / "tests" / "fixtures" / "face_reference.json").read_text()))


def test_colour_fixture_is_current():
    pytest.importorskip("PySide6")
    import json
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])  # noqa: F841
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_face_fixture
    assert make_face_fixture.make_colors() == json.loads((ROOT / "tests" / "fixtures" / "face_colors.json").read_text())


def test_web_imports_without_qt():
    """The web app must not pull in PySide6 (the Pages workflow installs only requirements-web.txt)."""
    code = ("import builtins, sys; real = builtins.__import__\n"
            "def imp(n, *a, **k):\n"
            "    if n.startswith('PySide6'): raise ImportError('blocked ' + n)\n"
            "    return real(n, *a, **k)\n"
            "builtins.__import__ = imp; sys.path.insert(0, 'src')\n"
            "import faceview.web.app")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_info_counts():
    info = client.get("/api/info").json()
    assert info["action_units"] == 12 and info["visemes"] == 15
    assert info["expressions"] == len(client.get("/api/expressions").json()["presets"])
    assert info["stylised_personas"] == sum(p["render_mode"] == "stylised" for p in client.get("/api/personas").json())


def test_speech_endpoint_matches_engine():
    from faceview.vision.speech import SpeechEngine
    r = client.post("/api/speech", json={"text": "Hello world.", "speed": 1.3}).json()
    tl = SpeechEngine().generate_au_sequence("Hello world.", speed=1.3)
    assert [t["viseme"] for t in r["timeline"]] == [t.viseme for t in tl]
    assert r["duration"] == pytest.approx(tl[-1].end_time)


def test_chat_without_key_is_503_and_calls_nothing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert client.get("/api/chat/status").json()["available"] is False
    r = client.post("/api/chat", json={"persona": "claude", "history": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503


def test_chat_status_never_returns_the_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    r = client.get("/api/chat/status")
    assert r.json()["available"] is True and "sk-test" not in r.text


def test_chat_conversation_is_trimmed_and_starts_with_user():
    from faceview.web.chat import MAX_TURNS, build_conversation
    hist = [{"role": "assistant" if i % 2 else "user", "content": f"m{i}"} for i in range(45)]
    msgs = build_conversation("claude", hist).for_anthropic()
    assert msgs[0]["role"] == "user" and len(msgs) <= MAX_TURNS and msgs[-1]["content"] == "m44"
    assert "Claude" in build_conversation("claude", []).effective_system()


def test_static_site_has_media_and_no_chat(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    site = tmp_path / "site"
    for g in client.get("/api/gallery").json():
        assert (site / "media" / g["file"]).exists()
    assert not (site / "api" / "chat.json").exists() and not (site / "api" / "chat").exists()
    for stem in client.get("/api/reference").json()["renders"]:
        assert (site / "static" / "ref" / f"{stem}.png").exists()
