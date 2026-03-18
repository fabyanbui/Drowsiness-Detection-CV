from __future__ import annotations

from pathlib import Path

from .utils import ensure_dir, utc_timestamp


def create_model_version_directory(
    models_dir: str | Path,
    version_name: str | None = None,
    default_prefix: str = "v1_baseline",
) -> tuple[str, Path]:
    root = ensure_dir(models_dir)
    resolved_name = version_name or f"{default_prefix}_{utc_timestamp()}"
    version_dir = ensure_dir(root / resolved_name)
    return resolved_name, version_dir

