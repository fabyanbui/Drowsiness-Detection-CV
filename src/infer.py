from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from .config import get_image_size, load_config
from .utils import (
    load_class_indices,
    probability_to_label,
    resolve_latest_model_path,
    utc_timestamp,
    write_json,
)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for drowsiness detection.")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to .keras model. If omitted, latest available model is used.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to single image file.")
    group.add_argument("--input-dir", help="Path to directory containing image files.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold for positive class probability.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional output path for JSON predictions.",
    )
    return parser.parse_args()


def _resolve_model_path(model_path: str | None, models_dir: str) -> Path:
    if model_path:
        resolved = Path(model_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Model path not found: {resolved}")
        return resolved
    return resolve_latest_model_path(models_dir)


def _preprocess_image(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = tf.keras.utils.load_img(image_path, target_size=image_size, color_mode="rgb")
    image_array = tf.keras.utils.img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


def _collect_input_paths(image: str | None, input_dir: str | None) -> list[Path]:
    if image:
        return [Path(image)]

    assert input_dir is not None
    directory = Path(input_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {directory}")

    image_paths = sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
    )
    if not image_paths:
        raise ValueError(f"No supported images found in directory: {directory}")
    return image_paths


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    paths_cfg = config["paths"]
    threshold = float(args.threshold or config["inference"]["default_threshold"])
    image_size = get_image_size(config)

    model_path = _resolve_model_path(
        model_path=args.model_path,
        models_dir=paths_cfg["models_dir"],
    )
    model = tf.keras.models.load_model(model_path)
    class_indices = load_class_indices(model_path.parent)

    input_paths = _collect_input_paths(args.image, args.input_dir)
    predictions: list[dict[str, Any]] = []
    for path in input_paths:
        batch = _preprocess_image(path, image_size=image_size)
        probability = float(model.predict(batch, verbose=0).reshape(-1)[0])
        predicted_label = probability_to_label(
            probability_of_positive_class=probability,
            class_indices=class_indices,
            threshold=threshold,
        )
        predictions.append(
            {
                "input_path": str(path),
                "predicted_label": predicted_label,
                "probability_of_positive_class": probability,
                "threshold": threshold,
            }
        )

    output_payload = {
        "model_path": str(model_path),
        "model_version": model_path.parent.name,
        "total_predictions": len(predictions),
        "predictions": predictions,
    }

    output_file = (
        Path(args.output_file)
        if args.output_file
        else Path(paths_cfg["metrics_dir"]) / f"inference_{utc_timestamp()}.json"
    )
    write_json(output_file, output_payload)

    return {
        "output_file": str(output_file),
        "predictions": predictions,
    }


def main() -> None:
    args = parse_args()
    result = run_inference(args)
    print("Inference complete.")
    print(f"- output_file: {result['output_file']}")
    print(f"- total_predictions: {len(result['predictions'])}")


if __name__ == "__main__":
    main()

