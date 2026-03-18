from __future__ import annotations

from typing import Any

from .utils import append_jsonl, utc_timestamp


def log_experiment_event(experiments_file: str, payload: dict[str, Any]) -> None:
    event = {
        "timestamp_utc": utc_timestamp(),
        **payload,
    }
    append_jsonl(experiments_file, event)

