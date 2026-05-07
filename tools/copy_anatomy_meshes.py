"""One-time copy of BodyParts3D head + neck STL meshes into the project.

faceView ships *without* the STL meshes (they're ~120 MB and licensed
separately by BodyParts3D). To enable the photo-anatomical render mode,
download a BodyParts3D dump and run this script pointing at its STL
directory::

    python -m tools.copy_anatomy_meshes /path/to/bodyparts3D/stl

The script copies the head + neck FMA subset into
``src/faceview/assets/anatomy_meshes/``. By default the destination
directory is git-ignored. Re-running is idempotent — files are skipped
if already present and identical.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from faceview.vision.anatomy_catalog import head_neck_fmas
from faceview.vision.anatomy_meshes import mesh_dir

HEAD_NECK_FMAS = head_neck_fmas()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Path to BodyParts3D stl/ directory")
    parser.add_argument("--all", action="store_true",
                         help="Copy all FMA-named STLs (~1.3 GB), not just head+neck")
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would be copied; don't actually copy")
    args = parser.parse_args()

    src = Path(args.source).expanduser().resolve()
    if not src.is_dir():
        print(f"ERROR: source directory not found: {src}", file=sys.stderr)
        return 2

    dest = mesh_dir()
    dest.mkdir(parents=True, exist_ok=True)

    if args.all:
        candidates = sorted(p for p in src.glob("*.stl"))
    else:
        candidates = []
        for fma in HEAD_NECK_FMAS:
            f = src / f"{fma}.stl"
            if f.exists():
                candidates.append(f)

    print(f"source: {src}")
    print(f"dest:   {dest}")
    print(f"will {'consider' if args.dry_run else 'copy'} {len(candidates)} STLs")

    copied = 0
    skipped = 0
    missing = 0
    for f in candidates:
        target = dest / f.name
        if target.exists() and target.stat().st_size == f.stat().st_size:
            skipped += 1
            continue
        if args.dry_run:
            print(f"  + {f.name}")
        else:
            shutil.copy2(f, target)
        copied += 1

    if not args.all:
        for fma in HEAD_NECK_FMAS:
            if not (src / f"{fma}.stl").exists():
                missing += 1
        if missing:
            print(f"  WARN: {missing} expected FMA STLs missing from source")

    action = "would copy" if args.dry_run else "copied"
    print(f"  {action} {copied}, skipped {skipped} (already present)")
    print(f"done. {len(list(dest.glob('*.stl')))} STLs now in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
