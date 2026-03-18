from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf

from .config import get_image_size, load_config
from .utils import (
    load_class_indices,
    probability_to_label,
    resolve_latest_model_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time inference simulation.")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to model file. If omitted, latest model is used.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source (webcam index like '0' or a video file path).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for positive class probability.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="Infer every N frames (defaults to config simulation.frame_stride).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process before exiting (defaults to config simulation.max_frames).",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/metrics/realtime_benchmark.json",
        help="Benchmark output JSON path.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display live preview window during simulation.",
    )
    return parser.parse_args()


def _resolve_capture_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def _prepare_frame(frame_bgr: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (image_size[1], image_size[0]))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def run_simulation(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    simulation_cfg = config["simulation"]
    image_size = get_image_size(config)

    model_path = (
        Path(args.model_path)
        if args.model_path
        else resolve_latest_model_path(config["paths"]["models_dir"])
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    class_indices = load_class_indices(model_path.parent)

    frame_stride = int(args.frame_stride or simulation_cfg["frame_stride"])
    max_frames = int(args.max_frames or simulation_cfg["max_frames"])
    if frame_stride <= 0:
        raise ValueError("frame-stride must be positive.")
    if max_frames <= 0:
        raise ValueError("max-frames must be positive.")

    capture_source = _resolve_capture_source(args.source)
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    inference_latencies_ms: list[float] = []
    inference_count = 0
    captured_frames = 0

    start_time = time.perf_counter()
    while captured_frames < max_frames:
        success, frame = capture.read()
        if not success:
            break

        captured_frames += 1
        if captured_frames % frame_stride != 0:
            continue

        preprocess_start = time.perf_counter()
        batch = _prepare_frame(frame, image_size=image_size)
        probability = float(model.predict(batch, verbose=0).reshape(-1)[0])
        predicted_label = probability_to_label(
            probability_of_positive_class=probability,
            class_indices=class_indices,
            threshold=args.threshold,
        )
        inference_latency_ms = (time.perf_counter() - preprocess_start) * 1000.0
        inference_latencies_ms.append(float(inference_latency_ms))
        inference_count += 1

        if args.display:
            label_text = f"{predicted_label} ({probability:.3f})"
            cv2.putText(
                frame,
                label_text,
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 255, 0),
                thickness=2,
            )
            cv2.imshow("Drowsiness Simulation", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    total_duration = time.perf_counter() - start_time
    capture.release()
    if args.display:
        cv2.destroyAllWindows()

    if not inference_latencies_ms:
        raise RuntimeError("No inference steps were executed during simulation.")

    benchmark = {
        "model_path": str(model_path),
        "video_source": str(args.source),
        "captured_frames": captured_frames,
        "inference_count": inference_count,
        "frame_stride": frame_stride,
        "total_duration_seconds": float(total_duration),
        "effective_fps": float(inference_count / total_duration) if total_duration > 0 else 0.0,
        "avg_inference_latency_ms": float(np.mean(inference_latencies_ms)),
        "p95_inference_latency_ms": float(np.percentile(inference_latencies_ms, 95)),
        "max_inference_latency_ms": float(np.max(inference_latencies_ms)),
    }

    output_path = Path(args.output_json)
    write_json(output_path, benchmark)
    return {"output_json": str(output_path), **benchmark}


def main() -> None:
    args = parse_args()
    result = run_simulation(args)
    print("Simulation complete.")
    print(f"- output_json: {result['output_json']}")
    print(f"- inference_count: {result['inference_count']}")
    print(f"- avg_inference_latency_ms: {result['avg_inference_latency_ms']:.3f}")
    print(f"- effective_fps: {result['effective_fps']:.3f}")


if __name__ == "__main__":
    main()

