"""Bake body_label_overrides_{male,female}.json into the
body_part_labels_{male,female}.npz files so runtime no longer needs
to apply them via _apply_manual_overrides. The JSON files are kept
for traceability (record of what changed and why) but a backup of
the original NPZ is written to assets/body_part_labels_<g>_orig.npz
the first time we bake (so we don't lose ground truth).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np


def main():
    from faceview.assets import assets_dir
    root = assets_dir()
    dry = "--dry" in sys.argv
    for g in ("male", "female"):
        npz_path = root / f"body_part_labels_{g}.npz"
        ov_path  = root / f"body_label_overrides_{g}.json"
        if not npz_path.exists():
            print(f"  skip {g}: no NPZ at {npz_path}")
            continue
        if not ov_path.exists():
            print(f"  skip {g}: no overrides at {ov_path}")
            continue

        backup = root / f"body_part_labels_{g}_orig.npz"
        if not backup.exists():
            shutil.copy2(npz_path, backup)
            print(f"  backed up → {backup.name}")

        data = dict(np.load(npz_path))
        labels = np.asarray(data["labels"], dtype=np.int32).copy()
        n = len(labels)

        ov = json.loads(ov_path.read_text())
        n_app = 0
        n_skip = 0
        for k, v in ov.items():
            if k.startswith("_"):  # metadata keys
                continue
            try:
                vi = int(k); lid = int(v)
            except (TypeError, ValueError):
                n_skip += 1
                continue
            if 0 <= vi < n and 0 <= lid <= 15:
                if labels[vi] != lid:
                    labels[vi] = lid
                    n_app += 1
            else:
                n_skip += 1

        data["labels"] = labels
        if dry:
            print(f"  {g}: would apply {n_app} overrides "
                    f"(skipped {n_skip})")
        else:
            np.savez(npz_path, **data)
            print(f"  {g}: baked {n_app} overrides into "
                    f"{npz_path.name} (skipped {n_skip})")


if __name__ == "__main__":
    main()
