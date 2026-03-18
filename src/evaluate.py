from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import load_config
from .data import build_test_generator
from .tracking import log_experiment_event
from .utils import (
    ensure_dir,
    index_to_label_map,
    load_class_indices,
    probability_to_class_index,
    resolve_latest_model_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to .keras model. If omitted, latest model is used.",
    )
    parser.add_argument(
        "--version-name",
        default=None,
        help="Model version directory name to evaluate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold for positive class probability.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional test batch size override.",
    )
    return parser.parse_args()


def _resolve_model_path(
    model_path: str | None, version_name: str | None, models_dir: str
) -> Path:
    if model_path:
        resolved = Path(model_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Model path not found: {resolved}")
        return resolved

    root = Path(models_dir)
    if version_name:
        version_dir = root / version_name
        best_candidate = version_dir / "model_best.keras"
        final_candidate = version_dir / "model_final.keras"
        if best_candidate.exists():
            return best_candidate
        if final_candidate.exists():
            return final_candidate
        raise FileNotFoundError(
            f"No model_best.keras or model_final.keras found in: {version_dir}"
        )

    return resolve_latest_model_path(root)


def _plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = int(matrix[row_index, col_index])
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="black",
            )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_roc_pr_curves(
    y_true_binary: np.ndarray,
    y_score_positive: np.ndarray,
    roc_pr_output_path: Path,
) -> tuple[float, float]:
    roc_auc = float(roc_auc_score(y_true_binary, y_score_positive))
    pr_auc = float(average_precision_score(y_true_binary, y_score_positive))

    fpr, tpr, _ = roc_curve(y_true_binary, y_score_positive)
    precision, recall, _ = precision_recall_curve(y_true_binary, y_score_positive)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(roc_pr_output_path)
    plt.close(fig)
    return roc_auc, pr_auc


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    paths_cfg = config["paths"]
    eval_cfg = config["evaluation"]

    model_path = _resolve_model_path(
        model_path=args.model_path,
        version_name=args.version_name,
        models_dir=paths_cfg["models_dir"],
    )
    model = tf.keras.models.load_model(model_path)
    model_version = model_path.parent.name

    class_indices = load_class_indices(model_path.parent)
    label_by_index = index_to_label_map(class_indices)
    sorted_indices = sorted(label_by_index.keys())
    negative_index = sorted_indices[0]
    positive_index = sorted_indices[1]
    threshold = float(args.threshold or eval_cfg["threshold"])

    test_generator = build_test_generator(
        config=config,
        batch_size=args.batch_size,
        seed=int(config["project"]["seed"]),
    )

    y_true = test_generator.classes.astype(int)
    y_scores = model.predict(test_generator, verbose=1).reshape(-1)

    if len(y_true) != len(y_scores):
        raise ValueError(
            "Prediction count does not match ground truth size: "
            f"{len(y_scores)} vs {len(y_true)}"
        )

    y_pred = np.array(
        [
            probability_to_class_index(
                probability_of_positive_class=float(score),
                class_indices=class_indices,
                threshold=threshold,
            )
            for score in y_scores
        ]
    )

    y_true_binary = (y_true == positive_index).astype(int)
    y_score_positive = y_scores if positive_index == 1 else (1.0 - y_scores)
    unique_classes = np.unique(y_true_binary)
    if len(unique_classes) != 2:
        raise ValueError(
            "Test labels must contain both classes for ROC-AUC/PR-AUC computation."
        )

    metrics_dir = ensure_dir(Path(paths_cfg["metrics_dir"]) / model_version)
    confusion_matrix_path = metrics_dir / "confusion_matrix.png"
    roc_pr_curve_path = metrics_dir / "roc_pr_curve.png"
    metrics_json_path = metrics_dir / "metrics.json"
    predictions_csv_path = metrics_dir / "predictions.csv"

    matrix = confusion_matrix(y_true, y_pred, labels=sorted_indices)
    _plot_confusion_matrix(
        matrix=matrix,
        labels=[label_by_index[index] for index in sorted_indices],
        output_path=confusion_matrix_path,
    )
    roc_auc, pr_auc = _plot_roc_pr_curves(
        y_true_binary=y_true_binary,
        y_score_positive=y_score_positive,
        roc_pr_output_path=roc_pr_curve_path,
    )

    metrics = {
        "model_path": str(model_path),
        "model_version": model_version,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=positive_index, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=positive_index, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, pos_label=positive_index, zero_division=0)
        ),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": matrix.tolist(),
        "class_indices": class_indices,
        "positive_class_label": label_by_index[positive_index],
        "negative_class_label": label_by_index[negative_index],
    }
    write_json(metrics_json_path, metrics)

    prediction_frame = pd.DataFrame(
        {
            "file_path": test_generator.filepaths,
            "y_true_index": y_true,
            "y_true_label": [label_by_index[index] for index in y_true],
            "y_score_positive_class": y_score_positive,
            "y_pred_index": y_pred,
            "y_pred_label": [label_by_index[index] for index in y_pred],
            "threshold": threshold,
        }
    )
    prediction_frame.to_csv(predictions_csv_path, index=False)

    log_experiment_event(
        experiments_file=paths_cfg["experiments_file"],
        payload={
            "event_type": "evaluation",
            "model_version": model_version,
            "model_path": str(model_path),
            "metrics_path": str(metrics_json_path),
            "predictions_path": str(predictions_csv_path),
            "metrics": {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            },
        },
    )

    return {
        "metrics_json_path": str(metrics_json_path),
        "predictions_csv_path": str(predictions_csv_path),
        "confusion_matrix_path": str(confusion_matrix_path),
        "roc_pr_curve_path": str(roc_pr_curve_path),
    }


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    print("Evaluation complete.")
    for key, value in result.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()

