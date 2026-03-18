from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

DEFAULT_CLASS_INDICES = {"DROWSY": 0, "NATURAL": 1}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if hasattr(tf.config.experimental, "enable_op_determinism"):
        tf.config.experimental.enable_op_determinism()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True))
        file.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_latest_model_path(models_dir: str | Path) -> Path:
    root = Path(models_dir)
    if not root.exists():
        raise FileNotFoundError(f"Models directory not found: {root}")

    preferred = sorted(
        root.glob("*/model_best.keras"),
        key=lambda model_path: model_path.stat().st_mtime,
        reverse=True,
    )
    if preferred:
        return preferred[0]

    fallback = sorted(
        root.glob("*/model_final.keras"),
        key=lambda model_path: model_path.stat().st_mtime,
        reverse=True,
    )
    if fallback:
        return fallback[0]

    raise FileNotFoundError(f"No model artifacts found in: {root}")


def load_class_indices(version_dir: str | Path) -> dict[str, int]:
    metadata_path = Path(version_dir) / "class_indices.json"
    if not metadata_path.exists():
        return DEFAULT_CLASS_INDICES.copy()

    raw = read_json(metadata_path)
    normalized = {str(label): int(index) for label, index in raw.items()}
    if len(normalized) != 2:
        raise ValueError(
            f"Expected binary class indices metadata, got {len(normalized)} classes."
        )
    return normalized


def index_to_label_map(class_indices: dict[str, int]) -> dict[int, str]:
    return {index: label for label, index in class_indices.items()}


def probability_to_class_index(
    probability_of_positive_class: float,
    class_indices: dict[str, int],
    threshold: float = 0.5,
) -> int:
    index_labels = index_to_label_map(class_indices)
    sorted_indices = sorted(index_labels)
    if len(sorted_indices) != 2:
        raise ValueError("Binary classification mapping is required.")

    negative_index = sorted_indices[0]
    positive_index = sorted_indices[1]
    return positive_index if probability_of_positive_class >= threshold else negative_index


def probability_to_label(
    probability_of_positive_class: float,
    class_indices: dict[str, int],
    threshold: float = 0.5,
) -> str:
    predicted_index = probability_to_class_index(
        probability_of_positive_class=probability_of_positive_class,
        class_indices=class_indices,
        threshold=threshold,
    )
    return index_to_label_map(class_indices)[predicted_index]

