"""Two-tier vision: ambient captioner + look-tool region/question."""

from __future__ import annotations

import numpy as np


def test_ambient_enabled_default_on(monkeypatch):
    from faceview.vision.scene_caption import ambient_vlm_enabled
    monkeypatch.delenv("FACEVIEW_AMBIENT_VLM", raising=False)
    assert ambient_vlm_enabled() is True


def test_ambient_enabled_disabled(monkeypatch):
    from faceview.vision.scene_caption import ambient_vlm_enabled
    monkeypatch.setenv("FACEVIEW_AMBIENT_VLM", "0")
    assert ambient_vlm_enabled() is False


def test_region_crop_named(monkeypatch):
    from faceview.llm.vision_tool import _crop_to_region
    img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    full = _crop_to_region(img, "full")
    assert full.shape == (100, 100, 3)
    tl = _crop_to_region(img, "top_left")
    assert tl.shape == (50, 50, 3)
    assert tl[0, 0, 0] == img[0, 0, 0]
    br = _crop_to_region(img, "bottom_right")
    assert br.shape == (50, 50, 3)
    assert br[0, 0, 0] == img[50, 50, 0]
    center = _crop_to_region(img, "center")
    assert center.shape == (50, 50, 3)
    # Unknown region falls back to full.
    weird = _crop_to_region(img, "left_diagonal_underbridge")
    assert weird.shape == (100, 100, 3)


def test_look_tool_schemas_accept_question_and_region():
    from faceview.llm.vision_tool import (
        LOOK_TOOL_ANTHROPIC, LOOK_TOOL_OLLAMA,
    )
    props = LOOK_TOOL_ANTHROPIC["input_schema"]["properties"]
    assert "question" in props
    assert "region" in props
    fprops = LOOK_TOOL_OLLAMA["function"]["parameters"]["properties"]
    assert "question" in fprops
    assert "region" in fprops


def test_pick_deep_vision_model_env_override(monkeypatch):
    from faceview.llm.vision_tool import pick_deep_vision_model
    monkeypatch.setenv("FACEVIEW_OLLAMA_DEEP_VISION_MODEL", "my-vlm:99b")
    assert pick_deep_vision_model() == "my-vlm:99b"


def test_pick_deep_vision_model_prefers_capable(monkeypatch):
    """Even if moondream is installed, the deep picker prefers a
    richer-captioning model."""
    import faceview.llm.vision_tool as vt
    monkeypatch.delenv("FACEVIEW_OLLAMA_DEEP_VISION_MODEL", raising=False)
    monkeypatch.setattr(
        "faceview.llm.ollama_client.list_ollama_models",
        lambda *_a, **_k: [
            "qwen2.5:14b", "moondream:latest", "llama3.2-vision:latest",
        ],
    )
    assert vt.pick_deep_vision_model() == "llama3.2-vision:latest"


def test_pick_deep_falls_back_to_moondream(monkeypatch):
    import faceview.llm.vision_tool as vt
    monkeypatch.delenv("FACEVIEW_OLLAMA_DEEP_VISION_MODEL", raising=False)
    monkeypatch.setattr(
        "faceview.llm.ollama_client.list_ollama_models",
        lambda *_a, **_k: ["qwen2.5:14b", "moondream:latest"],
    )
    assert vt.pick_deep_vision_model() == "moondream:latest"


def test_perception_narrate_includes_caption(fresh_bus, monkeypatch):
    """Scene captions surface in narrate_now and snapshot_dict with
    a longer freshness window than other signals."""
    import faceview.vision.perception as perc
    monkeypatch.setattr(perc.PerceptionStore, "_instance", None)
    store = perc.PerceptionStore.shared()
    from faceview.core.events import (
        EventType, Presence, SceneCaption,
    )
    fresh_bus.publish(EventType.PRESENCE, Presence(face_count=1, bboxes=[]))
    fresh_bus.publish(
        EventType.SCENE_CAPTION,
        SceneCaption(text="A person waving at the camera.",
                     model="moondream", latency_s=1.7),
    )
    snap = store.snapshot_dict()
    assert snap["scene_caption"]["text"] == "A person waving at the camera."
    text = store.narrate_now()
    assert "scene caption" in text
    assert "A person waving at the camera." in text
