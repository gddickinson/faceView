"""Per-concern lifecycle controllers for :class:`MainWindow`.

`MainWindow` used to be a 978-line god class that owned every worker
(camera, audio, STT, TTS, avatar, test mode, enrollment, monitor
windows) in addition to its proper job (laying out panels, building
menus). This package splits each concern into its own controller so
each file stays under the project's 500-line guideline and each
responsibility has a clear home.

The MainWindow facade keeps every external-facing method
(``set_camera_enabled``, ``set_persona``, …) and delegates to the
appropriate controller. External callers (``app.py``,
``server/service.py``, ``config_dialog.py``) don't see the change.
"""

from faceview.gui.controllers.audio_ctrl import AudioController
from faceview.gui.controllers.avatar_ctrl import AvatarController
from faceview.gui.controllers.camera_ctrl import CameraController
from faceview.gui.controllers.enrollment_ctrl import EnrollmentController
from faceview.gui.controllers.monitor_ctrl import MonitorController
from faceview.gui.controllers.test_mode_ctrl import TestModeController
from faceview.gui.controllers.tts_ctrl import TtsController

__all__ = [
    "AudioController",
    "AvatarController",
    "CameraController",
    "EnrollmentController",
    "MonitorController",
    "TestModeController",
    "TtsController",
]
