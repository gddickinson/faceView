"""P17 — `describe_room_layout` tool + RoomMapStore."""

from __future__ import annotations


def test_store_caches_latest_event(fresh_bus, monkeypatch):
    import faceview.vision.room_map as rm
    monkeypatch.setattr(rm.RoomMapStore, "_instance", None)
    store = rm.RoomMapStore.shared()
    from faceview.core.events import EventType, RoomMap, RoomMapItem
    assert store.latest() is None
    fresh_bus.publish(EventType.ROOM_MAP, RoomMap(items=[
        RoomMapItem(label="cup", x=0.3, z=1.2),
    ]))
    snap = store.latest()
    assert snap is not None
    assert len(snap.items) == 1
    assert snap.items[0].label == "cup"


def test_zone_classifier():
    from faceview.vision.room_map import _zone_for
    assert _zone_for(0.0, 1.0) == "directly ahead"
    assert _zone_for(0.7, 1.0) == "ahead and slightly to the right"
    assert _zone_for(1.5, 1.0) == "to the right"
    assert _zone_for(-0.7, 1.0) == "ahead and slightly to the left"
    assert _zone_for(-1.5, 1.0) == "to the left"
    assert _zone_for(0.0, -1.0) == "behind the camera"


def test_describe_no_map_returns_helpful(fresh_bus, monkeypatch):
    import faceview.vision.room_map as rm
    monkeypatch.setattr(rm.RoomMapStore, "_instance", None)
    rm.RoomMapStore.shared()
    msg = rm.describe_room_layout()
    assert "don't have a room map" in msg or "open the Room map" in msg


def test_describe_lists_items_in_distance_order(fresh_bus, monkeypatch):
    import faceview.vision.room_map as rm
    monkeypatch.setattr(rm.RoomMapStore, "_instance", None)
    rm.RoomMapStore.shared()
    from faceview.core.events import EventType, RoomMap, RoomMapItem
    fresh_bus.publish(EventType.ROOM_MAP, RoomMap(items=[
        RoomMapItem(label="laptop", x=0.0, z=3.0),       # far
        RoomMapItem(label="cup",    x=0.2, z=0.8),       # near
        RoomMapItem(label="person", x=-0.5, z=1.5),      # mid
    ]))
    msg = rm.describe_room_layout()
    # Near-to-far ordering means cup is mentioned before laptop.
    assert msg.index("cup") < msg.index("laptop")
    assert "ahead" in msg or "directly" in msg


def test_describe_truncates_after_six_items(fresh_bus, monkeypatch):
    import faceview.vision.room_map as rm
    monkeypatch.setattr(rm.RoomMapStore, "_instance", None)
    rm.RoomMapStore.shared()
    from faceview.core.events import EventType, RoomMap, RoomMapItem
    many = [RoomMapItem(label=f"item{i}", x=float(i), z=1.0)
            for i in range(10)]
    fresh_bus.publish(EventType.ROOM_MAP, RoomMap(items=many))
    msg = rm.describe_room_layout()
    assert "plus 4 other items" in msg


def test_describe_uses_metres_when_calibrated(fresh_bus, monkeypatch):
    import faceview.vision.room_map as rm
    monkeypatch.setattr(rm.RoomMapStore, "_instance", None)
    rm.RoomMapStore.shared()
    from faceview.core.events import EventType, RoomMap, RoomMapItem
    fresh_bus.publish(EventType.ROOM_MAP, RoomMap(
        items=[RoomMapItem(label="cup", x=0.0, z=1.0)],
        units="metres",
    ))
    msg = rm.describe_room_layout()
    assert "1.0 m" in msg


def test_tool_schemas_present():
    from faceview.llm.vision_tool import (
        DESCRIBE_ROOM_LAYOUT_TOOL_ANTHROPIC,
        DESCRIBE_ROOM_LAYOUT_TOOL_OLLAMA,
        TIER23_TOOLS_ANTHROPIC, TIER23_TOOLS_OLLAMA,
    )
    assert DESCRIBE_ROOM_LAYOUT_TOOL_ANTHROPIC["name"] == "describe_room_layout"
    assert (DESCRIBE_ROOM_LAYOUT_TOOL_OLLAMA["function"]["name"]
            == "describe_room_layout")
    # Should be in both bundles so engines surface it.
    names_a = {t["name"] for t in TIER23_TOOLS_ANTHROPIC}
    names_o = {t["function"]["name"] for t in TIER23_TOOLS_OLLAMA}
    assert "describe_room_layout" in names_a
    assert "describe_room_layout" in names_o


def test_tool_executor_routes_through_store(fresh_bus, monkeypatch):
    import faceview.vision.room_map as rm
    monkeypatch.setattr(rm.RoomMapStore, "_instance", None)
    rm.RoomMapStore.shared()
    from faceview.core.events import EventType, RoomMap, RoomMapItem
    fresh_bus.publish(EventType.ROOM_MAP, RoomMap(items=[
        RoomMapItem(label="cup", x=0.0, z=1.0),
    ]))
    from faceview.llm.vision_tool import run_describe_room_layout
    msg = run_describe_room_layout()
    assert "cup" in msg
