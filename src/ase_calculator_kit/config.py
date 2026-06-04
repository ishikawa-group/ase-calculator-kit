"""Configuration loading and resolution for config-driven calculators."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_CALCULATOR_ALIASES = {
    "espresso": "qe",
    "quantum-espresso": "qe",
    "qe": "qe",
    "vasp": "vasp",
}


def load_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load a calculator config from a YAML path or dictionary."""
    if isinstance(config, dict):
        return deepcopy(config)

    if isinstance(config, str | Path):
        path = Path(config)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file '{path}' must contain a YAML mapping.")
        return loaded

    raise TypeError("config must be a YAML path, pathlib.Path, or dictionary.")


def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Return ``base`` recursively merged with ``override``."""
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_calculator_config(
    name: str,
    *,
    config: str | Path | dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load, merge, and validate a calculator config."""
    if overrides is not None and not isinstance(overrides, dict):
        raise TypeError("overrides must be a dictionary when provided.")

    resolved = load_config(config)
    if overrides:
        resolved = deep_merge(resolved, overrides)

    requested = _canonical_calculator_name(name)
    configured_raw = resolved.get("calculator")
    if configured_raw is None:
        raise ValueError("Config requires calculator=.")
    if not isinstance(configured_raw, str):
        raise ValueError("Config calculator must be a string.")

    configured = _canonical_calculator_name(configured_raw)
    if configured != requested:
        raise ValueError(
            f"Config calculator='{configured_raw}' does not match requested "
            f"calculator '{name}'."
        )

    return resolved


def write_resolved_config_file(
    resolved: dict[str, Any],
    directory: str | Path,
) -> Path:
    """Write the final resolved config into the calculator working directory."""
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "resolved_calculator_config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)
    return path


def _canonical_calculator_name(name: str) -> str:
    key = name.lower()
    return _CALCULATOR_ALIASES.get(key, key)
