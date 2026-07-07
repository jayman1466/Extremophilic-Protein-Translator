"""Configuration loader.

Loads ``config/config.yaml`` into a nested dict with dotted-key access, and
resolves the repository root so paths work regardless of the caller's cwd.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Return the repository root (the dir containing config/ and src/)."""
    env = os.environ.get("EPTRANS_ROOT")
    if env:
        return Path(env).resolve()
    # src/eptrans/config.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def config_path() -> Path:
    return repo_root() / "config" / "config.yaml"


class Config(dict):
    """A dict subclass supporting dotted-key lookup: cfg.get_path('biotite.base')."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load and cache the project config."""
    p = Path(path) if path else config_path()
    with open(p) as fh:
        data = yaml.safe_load(fh)
    return Config(data)


if __name__ == "__main__":
    cfg = load_config()
    print(f"repo_root = {repo_root()}")
    print(f"gtdb release = {cfg.get_path('gtdb.release')}")
    print(f"biotite base = {cfg.get_path('biotite.base')}")
    print(f"thermophile_min_opt = {cfg.get_path('thresholds.temperature.thermophile_min_opt')}")
