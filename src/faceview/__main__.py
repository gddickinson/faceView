"""Entry point for ``python -m faceview`` / ``faceview`` console script."""

from __future__ import annotations

import sys

from faceview.app import main


if __name__ == "__main__":
    sys.exit(main())
