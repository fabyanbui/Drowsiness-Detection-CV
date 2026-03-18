from __future__ import annotations

import argparse

import numpy as np

from .config import get_input_shape, load_config
from .modeling import build_baseline_cnn
from .utils import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic training smoke test.")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path.")
    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help="Number of synthetic samples for smoke run.",
    )
    return parser.parse_args()


def run_smoke_test(args: argparse.Namespace) -> dict[str, float]:
    config = load_config(args.config)
    set_global_seed(int(config["project"]["seed"]))

    input_shape = get_input_shape(config)
    model = build_baseline_cnn(
        input_shape=input_shape,
        learning_rate=float(config["training"]["learning_rate"]),
    )

    if args.samples <= 0:
        raise ValueError("--samples must be positive.")

    x = np.random.rand(args.samples, *input_shape).astype(np.float32)
    y = np.random.randint(0, 2, size=(args.samples, 1)).astype(np.float32)

    model.fit(x, y, epochs=1, batch_size=min(8, args.samples), verbose=0)
    loss, accuracy = model.evaluate(x, y, verbose=0)
    return {"loss": float(loss), "accuracy": float(accuracy)}


def main() -> None:
    args = parse_args()
    result = run_smoke_test(args)
    print("Smoke test passed.")
    print(f"- loss: {result['loss']:.6f}")
    print(f"- accuracy: {result['accuracy']:.6f}")


if __name__ == "__main__":
    main()

