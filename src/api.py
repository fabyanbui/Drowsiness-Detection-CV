from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .config import get_image_size, load_config
from .monitoring import log_inference_event
from .utils import (
    load_class_indices,
    probability_to_label,
    resolve_latest_model_path,
    utc_timestamp,
)

app = FastAPI(title="Drowsiness Detection API", version="1.0.0")

APP_STATE: dict[str, Any] = {
    "config": None,
    "model": None,
    "model_path": None,
    "class_indices": None,
}


def _resolve_model_path(config: dict[str, Any]) -> Path:
    model_path_env = os.getenv("MODEL_PATH")
    if model_path_env:
        candidate = Path(model_path_env)
        if not candidate.exists():
            raise FileNotFoundError(f"MODEL_PATH does not exist: {candidate}")
        return candidate
    return resolve_latest_model_path(config["paths"]["models_dir"])


def _bytes_to_model_batch(payload: bytes, image_size: tuple[int, int]) -> np.ndarray:
    if not payload:
        raise ValueError("Uploaded file is empty.")
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc

    image = image.resize((image_size[1], image_size[0]))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


@app.on_event("startup")
def startup_event() -> None:
    config = load_config("configs/config.yaml")
    model_path = _resolve_model_path(config)
    model = tf.keras.models.load_model(model_path)
    class_indices = load_class_indices(model_path.parent)

    APP_STATE["config"] = config
    APP_STATE["model"] = model
    APP_STATE["model_path"] = model_path
    APP_STATE["class_indices"] = class_indices


@app.get("/health")
def health() -> dict[str, Any]:
    model_path = APP_STATE["model_path"]
    if model_path is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "ok", "model_path": str(model_path)}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = 0.5,
) -> dict[str, Any]:
    config = APP_STATE["config"]
    model = APP_STATE["model"]
    class_indices = APP_STATE["class_indices"]
    model_path = APP_STATE["model_path"]

    if config is None or model is None or class_indices is None or model_path is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    raw_content = await file.read()
    try:
        batch = _bytes_to_model_batch(raw_content, image_size=get_image_size(config))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    probability = float(model.predict(batch, verbose=0).reshape(-1)[0])
    predicted_label = probability_to_label(
        probability_of_positive_class=probability,
        class_indices=class_indices,
        threshold=threshold,
    )

    event = {
        "request_id": f"req_{utc_timestamp()}",
        "file_name": file.filename,
        "predicted_label": predicted_label,
        "probability_of_positive_class": probability,
        "threshold": threshold,
        "model_path": str(model_path),
    }
    log_inference_event(
        log_file=config["paths"]["monitoring_logs_file"],
        payload=event,
    )

    return event

