"""Bundled config and data files (FACS Action Units, expression presets,
CMU pronouncing dictionary subset). Adapted from the faceforge anatomy app's
animation system.
"""

from pathlib import Path


def assets_dir() -> Path:
    return Path(__file__).resolve().parent
