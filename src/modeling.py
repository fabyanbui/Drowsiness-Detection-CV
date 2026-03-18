from __future__ import annotations

from tensorflow.keras import layers, models, optimizers


def build_baseline_cnn(
    input_shape: tuple[int, int, int], learning_rate: float = 0.001
) -> models.Model:
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPool2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

