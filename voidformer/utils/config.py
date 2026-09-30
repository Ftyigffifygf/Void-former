"""Lightweight YAML config loader (Hydra-compatible structure, no runtime dep)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class Config(dict):
    """Dot-accessible nested dict."""

    def __getattr__(self, key: str) -> Any:
        try:
            v = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(v) if isinstance(v, Mapping) else v

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(data)


def save_config(cfg: Mapping, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)
