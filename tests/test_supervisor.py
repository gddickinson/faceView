"""R7 — worker-thread health supervisor."""

from __future__ import annotations

import threading
import time


def _make_thread(target, name="worker") -> threading.Thread:
    t = threading.Thread(target=target, name=name, daemon=True)
    t.start()
    return t


def test_register_and_status_alive():
    from faceview.core.supervisor import WorkerSupervisor
    WorkerSupervisor.reset_for_tests()
    sup = WorkerSupervisor.shared()
    # Long-running thread; will be alive when we check.
    stop = threading.Event()
    t = _make_thread(lambda: stop.wait(timeout=30))
    sup.register("alive_worker", t)
    snap = sup.status()
    assert "alive_worker" in snap
    assert snap["alive_worker"]["alive"] is True
    stop.set()
    WorkerSupervisor.reset_for_tests()


def test_thread_provider_for_threading_thread():
    from faceview.core.supervisor import _thread_provider_for
    t = threading.Thread(target=lambda: None, name="t1")
    fn = _thread_provider_for(t)
    assert fn() is t


def test_thread_provider_for_worker_with_underscore_thread():
    from faceview.core.supervisor import _thread_provider_for

    class _Worker:
        def __init__(self):
            self._thread = threading.Thread(target=lambda: None, name="t2")

    w = _Worker()
    fn = _thread_provider_for(w)
    assert fn() is w._thread


def test_thread_provider_for_unsupported_object_returns_none():
    from faceview.core.supervisor import _thread_provider_for
    fn = _thread_provider_for("not a worker")
    assert fn() is None


def test_unregister_removes_entry():
    from faceview.core.supervisor import WorkerSupervisor
    WorkerSupervisor.reset_for_tests()
    sup = WorkerSupervisor.shared()
    t = threading.Thread(target=lambda: None, name="t3")
    sup.register("temp", t)
    assert "temp" in sup.status()
    sup.unregister("temp")
    assert "temp" not in sup.status()
    WorkerSupervisor.reset_for_tests()


def test_dead_worker_publishes_status_and_invokes_restart(fresh_bus):
    """When a registered worker's thread isn't alive, _tick fires a
    STATUS event + the restart callback (capped at 3 attempts)."""
    from faceview.core.events import EventType
    from faceview.core.supervisor import WorkerSupervisor

    WorkerSupervisor.reset_for_tests()
    sup = WorkerSupervisor.shared()

    received: list = []
    fresh_bus.subscribe(EventType.STATUS, received.append)

    # Pre-died thread.
    t = threading.Thread(target=lambda: None, name="died")
    t.start()
    t.join()
    assert not t.is_alive()

    restarts = {"n": 0}

    def _restart():
        restarts["n"] += 1

    sup.register("died_worker", t, restart=_restart)
    # The first tick gives the worker a grace period (died_at = 0),
    # so it sets the died_at timestamp without publishing yet.
    sup._tick()
    # Second tick observes the worker is still dead and fires.
    sup._tick()
    # STATUS message announcing the death should have landed.
    assert any(
        getattr(p, "source", None) == "supervisor"
        and "died" in getattr(p, "message", "")
        for p in received
    )
    assert restarts["n"] == 1

    WorkerSupervisor.reset_for_tests()


def test_restart_capped_after_three_attempts(fresh_bus):
    from faceview.core.supervisor import WorkerSupervisor
    WorkerSupervisor.reset_for_tests()
    sup = WorkerSupervisor.shared()

    t = threading.Thread(target=lambda: None, name="died")
    t.start()
    t.join()
    restarts = {"n": 0}
    sup.register("died_worker", t, restart=lambda: restarts.__setitem__("n", restarts["n"] + 1))

    # Grace tick then up to 5 more — but only 3 restarts should fire.
    sup._tick()
    for _ in range(5):
        sup._tick()
    assert restarts["n"] == 3
    WorkerSupervisor.reset_for_tests()


def test_no_restart_callback_just_warns(fresh_bus):
    """When no restart is supplied, the supervisor still logs +
    publishes but doesn't crash."""
    from faceview.core.events import EventType
    from faceview.core.supervisor import WorkerSupervisor
    WorkerSupervisor.reset_for_tests()
    sup = WorkerSupervisor.shared()

    received: list = []
    fresh_bus.subscribe(EventType.STATUS, received.append)

    t = threading.Thread(target=lambda: None, name="died")
    t.start()
    t.join()
    sup.register("died_worker", t)  # no restart
    sup._tick()  # grace
    sup._tick()  # publishes
    assert any(getattr(p, "source", "") == "supervisor"
               for p in received)
    WorkerSupervisor.reset_for_tests()
