#!/usr/bin/env python3
"""Read-only monitor for a running faceView GUI.

Usage::

    python tools/faceview_monitor.py              # one-shot snapshot
    python tools/faceview_monitor.py status       # explicit alias
    python tools/faceview_monitor.py chat -n 30   # last 30 chat lines
    python tools/faceview_monitor.py events -n 50 # last 50 bus events
    python tools/faceview_monitor.py watch        # loop snapshot
    python tools/faceview_monitor.py screenshot path.png  # capture

Talks to the local control API (``127.0.0.1:8765`` by default). No auth
because the API only binds to loopback; Claude Code uses this to peek
at the running GUI without needing a screen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any


DEFAULT_HOST = "http://127.0.0.1:8765"


def _get(host: str, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{host}{path}"
    if params:
        from urllib.parse import urlencode
        url = f"{url}?{urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=2.5) as r:
            return json.loads(r.read())
    except urllib.error.URLError as exc:
        sys.stderr.write(f"faceView API not reachable on {host}: {exc.reason}\n")
        sys.exit(2)


def _post(host: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{host}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as r:
        return json.loads(r.read())


def _fmt_ts(ts: float) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _print_snapshot(snap: dict[str, Any]) -> None:
    eng = snap.get("engine") or "?"
    mdl = snap.get("model") or "—"
    print(f"╭─ faceView · {_fmt_ts(snap.get('ts', 0))} ───────────")
    print(f"│ Engine:   {eng}  ({mdl})")
    print(f"│ ANTH key: {'yes' if snap.get('anthropic_key') else 'no'}"
          f"     Persona: {snap.get('persona') or '—'}")
    workers = snap.get("workers") or {}
    line = "  ".join(
        f"{k}={'on' if v else 'off' if v is False else '?'}"
        for k, v in workers.items()
    )
    print(f"│ Workers:  {line}")
    test = snap.get("test") or {}
    if test.get("mode"):
        print(f"│ Test:     {test.get('engine')} / {test.get('model') or '—'} ({test['mode']})")
    cam = snap.get("camera_state") or {}
    pres = cam.get("presence", {}).get("face_count", 0)
    emo  = cam.get("emotion", {}).get("label", "—")
    mouth = "speaking" if cam.get("mouth", {}).get("speaking") else "silent"
    ident = cam.get("identity", {}).get("label", "—")
    print(f"│ Camera:   faces={pres}  emotion={emo}  mouth={mouth}  id={ident}")
    chat = snap.get("chat") or []
    if chat:
        print("│")
        print("│ Recent chat:")
        for line in chat[-8:]:
            who = (line.get("who") or "?")[:14]
            text = (line.get("text") or "").replace("\n", " ")
            if len(text) > 120:
                text = text[:117] + "…"
            print(f"│   [{_fmt_ts(line.get('ts', 0))}] {who:14s}  {text}")
    print("╰────────────────────────────────────────────")


def _print_chat(chat: list[dict[str, Any]]) -> None:
    for line in chat:
        who = line.get("who") or "?"
        text = (line.get("text") or "").rstrip()
        print(f"[{_fmt_ts(line.get('ts', 0))}] {who}: {text}")


def _print_events(events: list[dict[str, Any]]) -> None:
    for ev in events:
        et = ev.get("type", "?")
        payload = ev.get("payload")
        summary = ""
        if isinstance(payload, dict):
            keys = [k for k in payload if k not in ("ts",)]
            summary = " ".join(f"{k}={payload[k]!r}" for k in keys[:3])
        print(f"[{_fmt_ts(ev.get('ts', 0))}] {et:24s} {summary}")


def cmd_status(args) -> None:
    snap = _get(args.host, "/monitor", {"chat_n": args.chat_n, "events_n": args.events_n})
    if args.raw:
        print(json.dumps(snap, indent=2, default=str))
    else:
        _print_snapshot(snap)


def cmd_chat(args) -> None:
    data = _get(args.host, "/chat/log", {"n": args.n})
    _print_chat(data.get("chat") or [])


def cmd_events(args) -> None:
    events = _get(args.host, "/events", {"n": args.n})
    _print_events(events if isinstance(events, list) else [])


def cmd_watch(args) -> None:
    try:
        while True:
            snap = _get(args.host, "/monitor", {"chat_n": 8, "events_n": 0})
            # Clear screen between frames for terminal use.
            sys.stdout.write("\033[H\033[J")
            _print_snapshot(snap)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


def cmd_memory(args) -> None:
    data = _get(args.host, "/memory")
    if args.raw:
        print(json.dumps(data, indent=2, default=str))
        return
    if not data.get("ok"):
        print(json.dumps(data, indent=2))
        return
    rel = data.get("relationship") or {}
    emo = data.get("current_emotion") or ["neutral", 0]
    print(f"╭─ cognition · {data.get('persona')} ({data.get('character')}) ──")
    print(f"│ path:        {data.get('path')}")
    print(f"│ first_seen:  {data.get('first_seen')}   session #{data.get('session_count')}")
    print(f"│ user_name:   {data.get('user_name') or '—'}")
    print(f"│ relationship Lv {rel.get('level')} · {rel.get('name')}"
          f"  (score {rel.get('score')})")
    print(f"│ mood         {emo[0]} ({int((emo[1] or 0)*100)}%)")
    print(f"│ episodic     {data.get('episodic')} entries")
    print(f"│ semantic     subjects: {', '.join(data.get('semantic_subjects') or []) or '—'}")
    sem = (data.get("semantic") or {}).get("player") or {}
    if sem:
        print("│")
        print("│ known about player:")
        for k, v in list(sem.items())[-6:]:
            val = v.get('value') if isinstance(v, dict) else v
            print(f"│   {k[:22]:22s} {str(val)[:70]}")
    recent = (data.get("recent_episodic") or [])[-5:]
    if recent:
        print("│")
        print("│ recent episodic:")
        for m in recent:
            sig = m.get("significance", 0)
            emo = m.get("emotion", "neutral")
            print(f"│   sig={sig} {emo:11s} {m.get('text','')[:80]}")
    print("╰──────────────────────────────────────────")


def cmd_screenshot(args) -> None:
    resp = _post(args.host, "/screenshot", {"name": args.path, "encode_b64": False})
    if not resp.get("ok"):
        print(f"screenshot failed: {resp}", file=sys.stderr)
        sys.exit(1)
    print(resp.get("path", "(no path)"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only monitor for a running faceView GUI.")
    p.add_argument("--host", default=DEFAULT_HOST, help="Control API base URL")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("status", help="One-shot snapshot (default)")
    s.add_argument("--chat-n", type=int, default=8)
    s.add_argument("--events-n", type=int, default=0)
    s.add_argument("--raw", action="store_true", help="Print raw JSON")
    s.set_defaults(func=cmd_status)

    c = sub.add_parser("chat", help="Tail chat log")
    c.add_argument("-n", type=int, default=20)
    c.set_defaults(func=cmd_chat)

    e = sub.add_parser("events", help="Tail bus events")
    e.add_argument("-n", type=int, default=30)
    e.set_defaults(func=cmd_events)

    w = sub.add_parser("watch", help="Re-print the snapshot on an interval")
    w.add_argument("--interval", type=float, default=2.0)
    w.set_defaults(func=cmd_watch)

    mem = sub.add_parser("memory", help="Show the avatar's persistent memory store")
    mem.add_argument("--raw", action="store_true", help="Print raw JSON")
    mem.set_defaults(func=cmd_memory)

    sh = sub.add_parser("screenshot", help="Save a window screenshot via the API")
    sh.add_argument("path", help="Filename (e.g. shot.png)")
    sh.set_defaults(func=cmd_screenshot)

    return p


def main() -> None:
    parser = build_parser()
    # Default to `status` if no subcommand was given.
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1].startswith("-")):
        args = parser.parse_args(["status", *sys.argv[1:]])
    else:
        args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.parse_args(["status"]).func(parser.parse_args(["status"]))
        return
    args.func(args)


if __name__ == "__main__":
    main()
