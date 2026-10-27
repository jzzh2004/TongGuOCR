from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import load_config as load_pipeline_config


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the default config and optionally merge a user override."""

    config = load_pipeline_config(config_path)
    return config.get("blocking", config)
