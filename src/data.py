from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

from .config import get_image_size


@dataclass(frozen=True)
class DatasetLoaders:
    train: DirectoryIterator
    validation: DirectoryIterator
    test: DirectoryIterator


def _require_directory(path: str | Path, display_name: str) -> Path:
    directory = Path(path)
    if not directory.exists():
        raise FileNotFoundError(f"{display_name} directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"{display_name} path is not a directory: {directory}")
    return directory


def build_train_val_test_generators(
    config: dict[str, Any],
    batch_size: int | None = None,
    seed: int | None = None,
) -> DatasetLoaders:
    paths_config = config["paths"]
    training_config = config["training"]
    augmentation_config = training_config["augmentation"]

    train_dir = _require_directory(paths_config["train_dir"], "Train")
    test_dir = _require_directory(paths_config["test_dir"], "Test")

    effective_batch_size = int(batch_size or training_config["batch_size"])
    effective_seed = int(seed if seed is not None else config["project"]["seed"])
    validation_split = float(training_config["validation_split"])
    image_size = get_image_size(config)

    train_generator = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=validation_split,
        rotation_range=float(augmentation_config["rotation_range"]),
        width_shift_range=float(augmentation_config["width_shift_range"]),
        height_shift_range=float(augmentation_config["height_shift_range"]),
        shear_range=float(augmentation_config["shear_range"]),
        zoom_range=float(augmentation_config["zoom_range"]),
        horizontal_flip=bool(augmentation_config["horizontal_flip"]),
        fill_mode=str(augmentation_config["fill_mode"]),
    )
    eval_generator = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=validation_split,
    )
    test_generator = ImageDataGenerator(rescale=1.0 / 255.0)

    train_loader = train_generator.flow_from_directory(
        str(train_dir),
        target_size=image_size,
        class_mode="binary",
        batch_size=effective_batch_size,
        subset="training",
        shuffle=True,
        seed=effective_seed,
    )
    validation_loader = eval_generator.flow_from_directory(
        str(train_dir),
        target_size=image_size,
        class_mode="binary",
        batch_size=effective_batch_size,
        subset="validation",
        shuffle=False,
        seed=effective_seed,
    )
    test_loader = test_generator.flow_from_directory(
        str(test_dir),
        target_size=image_size,
        class_mode="binary",
        batch_size=effective_batch_size,
        shuffle=False,
        seed=effective_seed,
    )

    if train_loader.class_indices != validation_loader.class_indices:
        raise ValueError("Training and validation class index mappings do not match.")
    if train_loader.class_indices != test_loader.class_indices:
        raise ValueError("Training and test class index mappings do not match.")

    return DatasetLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
    )


def build_test_generator(
    config: dict[str, Any],
    batch_size: int | None = None,
    seed: int | None = None,
) -> DirectoryIterator:
    loaders = build_train_val_test_generators(
        config=config,
        batch_size=batch_size,
        seed=seed,
    )
    return loaders.test


def compute_steps(samples: int, batch_size: int, max_steps: int | None = None) -> int:
    if samples <= 0:
        raise ValueError("Samples must be positive.")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    default_steps = max(1, samples // batch_size)
    if max_steps is None:
        return default_steps
    return max(1, min(default_steps, int(max_steps)))

