from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf

from .config import get_image_size, load_config
from .monitoring import log_inference_event
from .utils import (
    load_class_indices,
    probability_to_label,
    resolve_latest_model_path,
    utc_timestamp,
    write_json,
)

DEFAULT_REALTIME_CONFIG: dict[str, Any] = {
    "default_threshold": 0.5,
    "frame_stride": 1,
    "alert_consecutive_frames": 3,
    "show_display": True,
    "emit_terminal_bell": False,
    "summary_output_dir": "artifacts/monitoring",
    "max_frames": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local real-time drowsiness detection from webcam/video."
    )
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
        default=None,
        help="Classification threshold override (0.0-1.0).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="Infer every N frames (default from config realtime.frame_stride).",
    )
    parser.add_argument(
        "--alert-consecutive-frames",
        type=int,
        default=None,
        help="Trigger alert after N consecutive DROWSY predictions.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on captured frames. If omitted, runs until quit/video end.",
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=30,
        help="Print status every N inference steps in no-display mode.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Monitoring JSONL path. Defaults to config paths.monitoring_logs_file.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary JSON output path.",
    )
    parser.add_argument(
        "--display",
        dest="display",
        action="store_true",
        help="Show OpenCV live preview window.",
    )
    parser.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        help="Disable OpenCV window (useful in headless environments).",
    )
    parser.set_defaults(display=None)
    parser.add_argument(
        "--emit-terminal-bell",
        dest="emit_terminal_bell",
        action="store_true",
        help="Emit terminal bell when alert is triggered.",
    )
    parser.add_argument(
        "--no-emit-terminal-bell",
        dest="emit_terminal_bell",
        action="store_false",
        help="Disable terminal bell on alert.",
    )
    parser.set_defaults(emit_terminal_bell=None)
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Validate model/source readiness and exit without running inference loop.",
    )
    return parser.parse_args()


def _resolve_capture_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def _resolve_model_path(config: dict[str, Any], model_path: str | None) -> Path:
    if model_path:
        candidate = Path(model_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Model path not found: {candidate}")
        return candidate
    return resolve_latest_model_path(config["paths"]["models_dir"])


def _prepare_frame(frame_bgr: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (image_size[1], image_size[0]))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def _merge_runtime_settings(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    realtime_cfg = {**DEFAULT_REALTIME_CONFIG, **(config.get("realtime") or {})}

    threshold = float(
        args.threshold
        if args.threshold is not None
        else realtime_cfg.get("default_threshold", config["inference"]["default_threshold"])
    )
    frame_stride = int(
        args.frame_stride
        if args.frame_stride is not None
        else realtime_cfg["frame_stride"]
    )
    alert_consecutive_frames = int(
        args.alert_consecutive_frames
        if args.alert_consecutive_frames is not None
        else realtime_cfg["alert_consecutive_frames"]
    )
    max_frames = (
        int(args.max_frames)
        if args.max_frames is not None
        else (
            int(realtime_cfg["max_frames"])
            if realtime_cfg.get("max_frames") is not None
            else None
        )
    )

    display = bool(
        args.display if args.display is not None else bool(realtime_cfg["show_display"])
    )
    emit_terminal_bell = bool(
        args.emit_terminal_bell
        if args.emit_terminal_bell is not None
        else bool(realtime_cfg["emit_terminal_bell"])
    )

    status_every = int(args.status_every)
    if status_every <= 0:
        raise ValueError("--status-every must be positive.")
    if frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")
    if alert_consecutive_frames <= 0:
        raise ValueError("--alert-consecutive-frames must be positive.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive when provided.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be in [0.0, 1.0].")

    summary_json = (
        Path(args.summary_json)
        if args.summary_json
        else Path(str(realtime_cfg["summary_output_dir"]))
        / f"realtime_session_{utc_timestamp()}.json"
    )
    log_file = Path(args.log_file) if args.log_file else Path(config["paths"]["monitoring_logs_file"])

    return {
        "threshold": threshold,
        "frame_stride": frame_stride,
        "alert_consecutive_frames": alert_consecutive_frames,
        "max_frames": max_frames,
        "display": display,
        "emit_terminal_bell": emit_terminal_bell,
        "status_every": status_every,
        "summary_json": summary_json,
        "log_file": log_file,
    }


def _build_health_payload(
    model_path: Path,
    source: str,
    resolved_source: int | str,
    threshold: float,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "mode": "local_realtime",
        "model_path": str(model_path),
        "source": source,
        "resolved_source": str(resolved_source),
        "threshold": threshold,
        "checked_at_utc": utc_timestamp(),
    }


def run_realtime_detection(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    image_size = get_image_size(config)
    model_path = _resolve_model_path(config=config, model_path=args.model_path)
    model = tf.keras.models.load_model(model_path)
    class_indices = load_class_indices(model_path.parent)
    settings = _merge_runtime_settings(config=config, args=args)

    capture_source = _resolve_capture_source(args.source)
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    health_payload = _build_health_payload(
        model_path=model_path,
        source=args.source,
        resolved_source=capture_source,
        threshold=settings["threshold"],
    )
    if args.health_check:
        capture.release()
        return health_payload

    start_utc = utc_timestamp()
    start_time = time.perf_counter()
    captured_frames = 0
    inference_count = 0
    drowsy_streak = 0
    alert_active = False
    alert_trigger_count = 0
    latencies_ms: list[float] = []

    latest_label = "N/A"
    latest_probability = 0.0
    latest_latency_ms = 0.0

    while True:
        success, frame = capture.read()
        if not success:
            break

        captured_frames += 1
        should_infer = captured_frames % settings["frame_stride"] == 0
        if should_infer:
            infer_started_at = time.perf_counter()
            batch = _prepare_frame(frame_bgr=frame, image_size=image_size)
            probability = float(model.predict(batch, verbose=0).reshape(-1)[0])
            predicted_label = probability_to_label(
                probability_of_positive_class=probability,
                class_indices=class_indices,
                threshold=settings["threshold"],
            )
            latency_ms = (time.perf_counter() - infer_started_at) * 1000.0

            latest_label = predicted_label
            latest_probability = probability
            latest_latency_ms = latency_ms

            inference_count += 1
            latencies_ms.append(latency_ms)

            is_drowsy = predicted_label.upper() == "DROWSY"
            drowsy_streak = drowsy_streak + 1 if is_drowsy else 0
            alert_just_triggered = (
                drowsy_streak >= settings["alert_consecutive_frames"] and not alert_active
            )
            alert_active = drowsy_streak >= settings["alert_consecutive_frames"]
            if alert_just_triggered:
                alert_trigger_count += 1
                if settings["emit_terminal_bell"]:
                    print("\a", end="", flush=True)

            event = {
                "request_id": f"rt_{utc_timestamp()}_{inference_count}",
                "source": str(args.source),
                "frame_index": captured_frames,
                "predicted_label": predicted_label,
                "probability_of_positive_class": probability,
                "threshold": settings["threshold"],
                "inference_latency_ms": latency_ms,
                "alert_active": alert_active,
                "drowsy_streak": drowsy_streak,
                "alert_just_triggered": alert_just_triggered,
                "mode": "local_realtime",
                "model_path": str(model_path),
            }
            log_inference_event(log_file=settings["log_file"], payload=event)

            if not settings["display"] and inference_count % settings["status_every"] == 0:
                print(
                    "[realtime] "
                    f"frames={captured_frames} "
                    f"inferences={inference_count} "
                    f"label={predicted_label} "
                    f"prob={probability:.3f} "
                    f"latency_ms={latency_ms:.2f} "
                    f"alert={alert_active}"
                )

        if settings["display"]:
            elapsed_seconds = time.perf_counter() - start_time
            effective_fps = (
                float(inference_count / elapsed_seconds) if elapsed_seconds > 0 else 0.0
            )
            status_color = (0, 0, 255) if alert_active else (0, 255, 0)
            status_text = f"{latest_label} ({latest_probability:.3f})"
            details_text = (
                f"lat={latest_latency_ms:.1f}ms "
                f"fps={effective_fps:.2f} "
                f"streak={drowsy_streak}"
            )
            alert_text = "ALERT: DROWSINESS DETECTED" if alert_active else "Status: OK"

            cv2.putText(
                frame,
                status_text,
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.9,
                color=status_color,
                thickness=2,
            )
            cv2.putText(
                frame,
                details_text,
                org=(20, 75),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(255, 255, 255),
                thickness=2,
            )
            cv2.putText(
                frame,
                alert_text,
                org=(20, 110),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=status_color,
                thickness=2,
            )
            cv2.putText(
                frame,
                "Press 'q' to quit",
                org=(20, 145),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(200, 200, 200),
                thickness=1,
            )

            cv2.imshow("Drowsiness Realtime Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if settings["max_frames"] is not None and captured_frames >= settings["max_frames"]:
            break

    total_duration_seconds = time.perf_counter() - start_time
    capture.release()
    if settings["display"]:
        cv2.destroyAllWindows()

    if inference_count == 0:
        raise RuntimeError(
            "No inference steps were executed. Check source, frame-stride, and max-frames."
        )

    summary = {
        "status": "ok",
        "mode": "local_realtime",
        "started_at_utc": start_utc,
        "finished_at_utc": utc_timestamp(),
        "model_path": str(model_path),
        "source": str(args.source),
        "captured_frames": captured_frames,
        "inference_count": inference_count,
        "threshold": settings["threshold"],
        "frame_stride": settings["frame_stride"],
        "alert_consecutive_frames": settings["alert_consecutive_frames"],
        "alert_trigger_count": alert_trigger_count,
        "log_file": str(settings["log_file"]),
        "summary_json": str(settings["summary_json"]),
        "total_duration_seconds": float(total_duration_seconds),
        "effective_fps": float(inference_count / total_duration_seconds)
        if total_duration_seconds > 0
        else 0.0,
        "avg_inference_latency_ms": float(np.mean(latencies_ms)),
        "p95_inference_latency_ms": float(np.percentile(latencies_ms, 95)),
        "max_inference_latency_ms": float(np.max(latencies_ms)),
    }
    write_json(settings["summary_json"], summary)
    return summary


def main() -> None:
    args = parse_args()
    result = run_realtime_detection(args)
    if args.health_check:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print("Realtime detection session complete.")
    print(f"- summary_json: {result['summary_json']}")
    print(f"- captured_frames: {result['captured_frames']}")
    print(f"- inference_count: {result['inference_count']}")
    print(f"- effective_fps: {result['effective_fps']:.3f}")
    print(f"- avg_inference_latency_ms: {result['avg_inference_latency_ms']:.3f}")
    print(f"- alert_trigger_count: {result['alert_trigger_count']}")


if __name__ == "__main__":
    main()

