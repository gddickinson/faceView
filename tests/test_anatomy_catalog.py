"""Catalog: faceforge head-anatomy configs unified into MeshSpec records."""

from __future__ import annotations

from faceview.vision.anatomy_catalog import (
    head_neck_fmas,
    load_catalog,
    specs_by_category,
    specs_for_layer_set,
)


def test_catalog_loads_all_categories():
    cats = specs_by_category()
    assert "bone" in cats and len(cats["bone"]) >= 15
    assert "muscle" in cats and len(cats["muscle"]) >= 80
    assert "skin" in cats and len(cats["skin"]) == 1


def test_head_neck_fmas_is_dict_of_fma_to_name():
    fmas = head_neck_fmas()
    assert all(k.startswith("FMA") for k in fmas)
    assert all(isinstance(v, str) and v for v in fmas.values())
    assert len(fmas) >= 100


def test_layer_set_skull_only_excludes_muscles():
    specs = specs_for_layer_set("skull_only")
    cats = {s.category for s in specs}
    assert "muscle" not in cats
    assert "bone" in cats


def test_layer_set_lifelike_makes_skin_opaque():
    specs = specs_for_layer_set("lifelike")
    skins = [s for s in specs if s.category == "skin"]
    assert skins, "expected a skin spec"
    assert skins[0].opacity > 0.85


def test_layer_set_xray_uses_translucent_skin():
    specs = specs_for_layer_set("xray")
    skins = [s for s in specs if s.category == "skin"]
    assert skins
    assert skins[0].opacity < 0.6


def test_specs_have_color_tuples():
    for spec in load_catalog():
        assert len(spec.color) == 3
        for c in spec.color:
            assert 0 <= c <= 255


def test_unknown_layer_set_raises():
    import pytest
    with pytest.raises(ValueError):
        specs_for_layer_set("not_a_real_set")
