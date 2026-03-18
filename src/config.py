from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    required_sections = ("project", "paths", "training", "callbacks", "evaluation")
    for section in required_sections:
        if section not in loaded:
            raise KeyError(f"Missing required config section: '{section}'")
    return loaded


def get_image_size(config: dict[str, Any]) -> tuple[int, int]:
    training = config["training"]
    return int(training["image_height"]), int(training["image_width"])


def get_input_shape(config: dict[str, Any]) -> tuple[int, int, int]:
    image_height, image_width = get_image_size(config)
    channels = int(config["training"]["channels"])
    return image_height, image_width, channels

