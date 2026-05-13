#!/usr/bin/env python3
"""Drive a (running or not yet running) faceView GUI from the CLI.

Read-only inspection lives in ``tools/faceview_monitor.py``; this one
writes — it can also start the GUI itself, pulling the Anthropic API
key from the macOS Keychain entry the user already set up.

Subcommands::

    launch [--test ENGINE] [--test-model M] [--no-key]   start the GUI
    stop                                                  ask the GUI to close
    chat "text"                                           send a user message
    say "text"                                            avatar mouths text
    persona NAME                                          swap avatar persona
    emotion NAME                                          set avatar mood
    engine NAME [--model M]                               swap the LLM engine
    test ENGINE [--model M]                               configure test bots
    lifecycle NAME --on / --off                           flip a worker
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


DEFAULT_HOST = "http://127.0.0.1:8765"
FACEVIEW_BIN = "/opt/anaconda3/envs/faceview/bin/faceview"


def _post(host: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{host}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8.0) as r:
            return json.loads(r.read())
    except urllib.error.URLError as exc:
        sys.stderr.write(f"faceView API not reachable on {host}: {exc.reason}\n")
        sys.exit(2)


def _get(host: str, path: str) -> dict:
    try:
        with urllib.request.urlopen(f"{host}{path}", timeout=2.5) as r:
            return json.loads(r.read())
    except urllib.error.URLError:
        return {}


def _is_running(host: str) -> bool:
    return bool(_get(host, "/healthz").get("ok"))


def _read_keychain_key() -> str | None:
    try:
        out = subprocess.check_output(
            ["security", "find-generic-password",
             "-a", os.environ.get("USER", ""),
             "-s", "ANTHROPIC_API_KEY", "-w"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() or None
    except Exception:  # noqa: BLE001
        return None


# ── launch / stop ────────────────────────────────────────────────────


def cmd_launch(args) -> None:
    if _is_running(args.host):
        print(f"faceView already running at {args.host}")
        return
    env = os.environ.copy()
    if not args.no_key and "ANTHROPIC_API_KEY" not in env:
        key = _read_keychain_key()
        if key:
            env["ANTHROPIC_API_KEY"] = key
    if args.test:
        env["FACEVIEW_TEST_MODE"] = "1"
        env["FACEVIEW_TEST_ENGINE"] = args.test
        if args.test_model:
            env["FACEVIEW_TEST_MODEL"] = args.test_model
    if args.persona:
        env["FACEVIEW_AVATAR_PERSONA"] = args.persona
    bin_path = args.bin or FACEVIEW_BIN
    if not os.path.exists(bin_path):
        sys.stderr.write(f"faceview binary not found at {bin_path}\n")
        sys.exit(2)
    # Detach so the CLI returns immediately. Output goes to /dev/null
    # because the user's running this from a Claude session.
    proc = subprocess.Popen(
        [bin_path],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Poll until the API is up so the CLI can chain commands.
    deadline = time.time() + (args.wait or 25)
    while time.time() < deadline:
        if _is_running(args.host):
            print(f"faceView up (pid {proc.pid})")
            return
        time.sleep(0.5)
    sys.stderr.write(
        f"launched pid {proc.pid}, but API didn't come up in {args.wait}s\n"
    )
    sys.exit(3)


def cmd_stop(args) -> None:
    if not _is_running(args.host):
        print("faceView not running")
        return
    resp = _post(args.host, "/shutdown", {})
    if not resp.get("ok"):
        sys.stderr.write(f"shutdown failed: {resp}\n")
        sys.exit(1)
    print("shutdown queued")


# ── chat / avatar / engine / lifecycle ────────────────────────────────


def cmd_chat(args) -> None:
    print(json.dumps(_post(args.host, "/chat", {"text": args.text}), indent=2))


def cmd_say(args) -> None:
    print(json.dumps(_post(args.host, "/avatar/say",
                           {"text": args.text, "speed": args.speed}), indent=2))


def cmd_persona(args) -> None:
    print(json.dumps(_post(args.host, "/avatar/persona", {"name": args.name}), indent=2))


def cmd_emotion(args) -> None:
    print(json.dumps(_post(args.host, "/avatar/emotion", {"name": args.name}), indent=2))


def cmd_engine(args) -> None:
    body: dict = {"engine": args.name}
    if args.model:
        body["model"] = args.model
    print(json.dumps(_post(args.host, "/llm/engine", body), indent=2))


def cmd_test(args) -> None:
    body: dict = {"engine": args.engine}
    if args.model:
        body["model"] = args.model
    print(json.dumps(_post(args.host, "/test/engine", body), indent=2))


def cmd_lifecycle(args) -> None:
    on = True if args.on else False if args.off else None
    if on is None:
        sys.stderr.write("specify --on or --off\n")
        sys.exit(2)
    print(json.dumps(_post(args.host, "/lifecycle",
                           {"name": args.name, "on": on}), indent=2))


# ── arg parser ───────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default=DEFAULT_HOST)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("launch", help="Start the GUI (idempotent)")
    sp.add_argument("--test", choices=("canned", "ollama", "anthropic", "demo"))
    sp.add_argument("--test-model", default=None)
    sp.add_argument("--persona", default=None)
    sp.add_argument("--bin", default=None, help="Path to the faceview binary")
    sp.add_argument("--wait", type=float, default=25.0,
                    help="Seconds to wait for the API to come up")
    sp.add_argument("--no-key", action="store_true",
                    help="Don't pull ANTHROPIC_API_KEY from Keychain")
    sp.set_defaults(func=cmd_launch)

    sp = sub.add_parser("stop", help="Ask the GUI to close")
    sp.set_defaults(func=cmd_stop)

    sp = sub.add_parser("chat", help="Send a user-side chat message")
    sp.add_argument("text")
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("say", help="Drive the avatar to mouth text (visemes only)")
    sp.add_argument("text")
    sp.add_argument("--speed", type=float, default=1.0)
    sp.set_defaults(func=cmd_say)

    sp = sub.add_parser("persona", help="Swap avatar persona")
    sp.add_argument("name")
    sp.set_defaults(func=cmd_persona)

    sp = sub.add_parser("emotion", help="Set avatar baseline emotion")
    sp.add_argument("name")
    sp.set_defaults(func=cmd_emotion)

    sp = sub.add_parser("engine", help="Live-swap the main LLM engine")
    sp.add_argument("name", choices=("auto", "anthropic", "ollama", "demo"))
    sp.add_argument("--model", default=None)
    sp.set_defaults(func=cmd_engine)

    sp = sub.add_parser("test", help="Configure test-mode bot engine")
    sp.add_argument("engine", choices=("canned", "ollama", "anthropic", "demo"))
    sp.add_argument("--model", default=None)
    sp.set_defaults(func=cmd_test)

    sp = sub.add_parser("lifecycle", help="Toggle a worker on/off")
    sp.add_argument("name",
                    choices=("camera", "mic", "tts", "avatar", "test_mode", "mirror"))
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--on", action="store_true")
    g.add_argument("--off", action="store_true")
    sp.set_defaults(func=cmd_lifecycle)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
