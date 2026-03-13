"""
Runtime config helpers for CLI and UI flows.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

from config import config_loader


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
EXAMPLE_CONFIG_PATH = CONFIG_DIR / "config.example.yaml"


def load_effective_config() -> Dict[str, Any]:
    """Load the current config file, falling back to the tracked example config."""
    source_path = CONFIG_PATH if CONFIG_PATH.exists() else EXAMPLE_CONFIG_PATH
    with open(source_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def save_config(config: Dict[str, Any]) -> Path:
    """Persist the user config and return its path."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    return CONFIG_PATH


def apply_runtime_config(config: Dict[str, Any]):
    """
    Mutate the shared CONFIG object in-place so modules importing CONFIG keep seeing
    the refreshed values without requiring a process restart.
    """
    config_loader.CONFIG.clear()
    config_loader.CONFIG.update(deepcopy(config))


def load_and_apply_config() -> Dict[str, Any]:
    """Convenience helper for UI flows."""
    config = load_effective_config()
    apply_runtime_config(config)
    return config


def mask_secret(value: str, *, visible_tail: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= visible_tail:
        return "*" * len(value)
    return f"{'*' * (len(value) - visible_tail)}{value[-visible_tail:]}"
