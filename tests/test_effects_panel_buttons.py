"""Every effect button in the Effects panel is either wired to a real handler or disabled.

Regression: the Body tab offered pose/limb effects (arms_up, wave_left, body_lean_left, ...)
with no EffectSpec and no handler, so clicking them silently did nothing.
"""
from __future__ import annotations

from faceview.vision.effects import (
    POST_HANDLERS, PRE_HANDLERS, REGISTRY, Stage, is_implemented,
)


def test_every_registered_effect_has_a_handler():
    for spec in REGISTRY:
        handlers = PRE_HANDLERS if spec.stage == Stage.PRE else POST_HANDLERS
        assert spec.name in handlers, spec.name
        assert is_implemented(spec.name)


def test_unknown_effect_is_not_implemented():
    assert not is_implemented("arms_up_that_does_not_exist")


def test_panel_buttons_enabled_only_with_a_handler(app):
    from PySide6.QtWidgets import QPushButton
    from faceview.gui.effects_panel import EffectsPanel
    panel = EffectsPanel()
    buttons = [b for b in panel.findChildren(QPushButton) if b.property("effect_name")]
    assert len(buttons) >= len(REGISTRY)
    for b in buttons:
        name = b.property("effect_name")
        assert b.isEnabled() == is_implemented(name), name
        if not b.isEnabled():
            assert "not implemented" in b.toolTip().lower(), name
    panel.deleteLater()
