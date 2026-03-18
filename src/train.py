from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from .config import get_input_shape, load_config
from .data import build_train_val_test_generators, compute_steps
from .modeling import build_baseline_cnn
from .tracking import log_experiment_event
from .utils import set_global_seed, write_json
from .versioning import create_model_version_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train drowsiness detection CNN model.")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override training batch size."
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--version-name",
        type=str,
        default=None,
        help="Model version name (e.g. v1_baseline).",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Cap train steps per epoch (useful for smoke runs).",
    )
    parser.add_argument(
        "--max-val-steps",
        type=int,
        default=None,
        help="Cap validation steps (useful for smoke runs).",
    )
    parser.add_argument(
        "--default-prefix",
        type=str,
        default="v1_baseline",
        help="Default version prefix when version name is not provided.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)

    training_cfg = config["training"]
    paths_cfg = config["paths"]
    callbacks_cfg = config["callbacks"]

    seed = int(args.seed if args.seed is not None else config["project"]["seed"])
    batch_size = int(args.batch_size or training_cfg["batch_size"])
    epochs = int(args.epochs or training_cfg["epochs"])

    set_global_seed(seed)

    loaders = build_train_val_test_generators(
        config=config,
        batch_size=batch_size,
        seed=seed,
    )

    input_shape = get_input_shape(config)
    learning_rate = float(training_cfg["learning_rate"])
    model = build_baseline_cnn(input_shape=input_shape, learning_rate=learning_rate)

    version_name, version_dir = create_model_version_directory(
        models_dir=paths_cfg["models_dir"],
        version_name=args.version_name,
        default_prefix=args.default_prefix,
    )

    best_model_path = version_dir / "model_best.keras"
    final_model_path = version_dir / "model_final.keras"
    history_path = version_dir / "history.json"
    class_indices_path = version_dir / "class_indices.json"
    metadata_path = version_dir / "training_metadata.json"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=int(callbacks_cfg["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(callbacks_cfg["reduce_lr_factor"]),
            patience=int(callbacks_cfg["reduce_lr_patience"]),
            min_lr=float(callbacks_cfg["min_lr"]),
        ),
    ]

    train_steps = compute_steps(
        samples=loaders.train.samples,
        batch_size=loaders.train.batch_size,
        max_steps=args.max_train_steps,
    )
    val_steps = compute_steps(
        samples=loaders.validation.samples,
        batch_size=loaders.validation.batch_size,
        max_steps=args.max_val_steps,
    )

    history = model.fit(
        loaders.train,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=loaders.validation,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(final_model_path)

    write_json(
        history_path,
        {
            "epoch": [int(epoch) for epoch in history.epoch],
            "history": history.history,
        },
    )
    write_json(class_indices_path, loaders.train.class_indices)

    training_metadata = {
        "version_name": version_name,
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "train_samples": int(loaders.train.samples),
        "validation_samples": int(loaders.validation.samples),
        "test_samples": int(loaders.test.samples),
        "train_steps": train_steps,
        "validation_steps": val_steps,
        "input_shape": input_shape,
        "learning_rate": learning_rate,
        "model_best_path": str(best_model_path),
        "model_final_path": str(final_model_path),
    }
    write_json(metadata_path, training_metadata)

    log_experiment_event(
        experiments_file=paths_cfg["experiments_file"],
        payload={
            "event_type": "training",
            "version_name": version_name,
            "model_final_path": str(final_model_path),
            "model_best_path": str(best_model_path),
            "metrics": {
                "final_train_accuracy": float(history.history["accuracy"][-1]),
                "final_val_accuracy": float(history.history["val_accuracy"][-1]),
                "final_train_loss": float(history.history["loss"][-1]),
                "final_val_loss": float(history.history["val_loss"][-1]),
            },
            "parameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
                "learning_rate": learning_rate,
            },
        },
    )

    return {
        "version_name": version_name,
        "model_final_path": str(final_model_path),
        "model_best_path": str(best_model_path),
        "history_path": str(history_path),
        "metadata_path": str(metadata_path),
    }


def main() -> None:
    args = parse_args()
    result = train(args)
    print("Training complete.")
    for key, value in result.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()

