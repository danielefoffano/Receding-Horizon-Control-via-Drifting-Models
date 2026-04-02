from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a mutable dictionary."""
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def clone_config(config: dict[str, Any]) -> dict[str, Any]:
    """Create a deep copy so runtime overrides do not mutate the caller's view."""
    return copy.deepcopy(config)


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if base_dir is None:
        return candidate.resolve()
    return (Path(base_dir).expanduser() / candidate).resolve()


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def dump_config(config: dict[str, Any], path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return target
