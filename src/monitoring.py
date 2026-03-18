from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .config import load_config
from .utils import append_jsonl, read_jsonl, utc_timestamp, write_json


def log_inference_event(log_file: str | Path, payload: dict[str, Any]) -> None:
    event = {
        "timestamp_utc": utc_timestamp(),
        **payload,
    }
    append_jsonl(log_file, event)


def summarize_inference_logs(
    log_file: str | Path,
    output_file: str | Path | None = None,
) -> dict[str, Any]:
    records = read_jsonl(log_file)
    if not records:
        raise ValueError(f"No inference records found in {log_file}")

    labels = [record["predicted_label"] for record in records]
    probabilities = [float(record["probability_of_positive_class"]) for record in records]
    class_counter = Counter(labels)

    summary = {
        "log_file": str(log_file),
        "total_predictions": len(records),
        "prediction_distribution": dict(class_counter),
        "avg_probability_of_positive_class": float(sum(probabilities) / len(probabilities)),
        "min_probability_of_positive_class": float(min(probabilities)),
        "max_probability_of_positive_class": float(max(probabilities)),
    }

    if output_file is not None:
        write_json(output_file, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize inference monitoring logs.")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path.")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Inference log JSONL file path. Defaults to config path.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Summary JSON output path. Defaults to artifacts/monitoring/summary_<timestamp>.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    log_file = args.log_file or config["paths"]["monitoring_logs_file"]
    output_file = (
        args.output_file
        or Path("artifacts/monitoring") / f"summary_{utc_timestamp()}.json"
    )
    summary = summarize_inference_logs(log_file=log_file, output_file=output_file)
    print("Monitoring summary complete.")
    print(f"- output_file: {output_file}")
    print(f"- total_predictions: {summary['total_predictions']}")


if __name__ == "__main__":
    main()

