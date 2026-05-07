"""Tests for openFACS bridge + MediaPipe capture stubs."""

from __future__ import annotations

import json
import socket

import pytest


def test_openfacs_payload_round_trip():
    """OpenFACS bridge sends well-formed JSON over UDP."""
    from faceview.vision.openfacs_bridge import OpenFACSBridge

    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]

    bridge = OpenFACSBridge(port=port)
    bridge.send({"AU12": 0.8, "AU26": 0.4})
    server.settimeout(1.0)
    msg, _ = server.recvfrom(2048)
    payload = json.loads(msg.decode("ascii"))
    assert payload["AU12"] == 0.8
    assert payload["AU26"] == 0.4
    assert "speed" in payload
    bridge.close()
    server.close()


def test_mediapipe_capture_import_does_not_raise():
    """The module should be importable even without mediapipe installed."""
    from faceview.vision import mediapipe_capture  # noqa: F401


def test_mediapipe_capture_module_imports():
    """The module imports cleanly without mediapipe installed."""
    from faceview.vision import mediapipe_capture
    # Has the public class.
    assert hasattr(mediapipe_capture, "MediaPipeCapture")


def test_decimated_gpu_path_routes_when_meshes_present(qtbot):
    from faceview.vision.anatomy_meshes import meshes_available
    if not meshes_available():
        pytest.skip("BP3D STLs not present")
    pytest.importorskip("moderngl")
    from faceview.vision.sim_face import FaceParams, render_face
    p = FaceParams.neutral()
    p.render_mode = "head_decimated_3d_gpu"
    frame = render_face(p, (200, 200))
    assert frame.shape == (200, 200, 3)
