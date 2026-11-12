"""Start the faceView web app on http://127.0.0.1:8199 (local only).

    pip install -r requirements-web.txt
    python run_web.py [--port 8199]

Live chat (the Chat page) needs ANTHROPIC_API_KEY in this process's environment and `pip install anthropic`;
the model is FACEVIEW_MODEL (default as in src/faceview/config.py). Everything else works without a key.
"""
import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8199)
    a = ap.parse_args()
    print(f"faceView web app: http://127.0.0.1:{a.port}")
    uvicorn.run("faceview.web.app:app", host="127.0.0.1", port=a.port)
