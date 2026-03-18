# MLOps Review Checkpoints

This document lists decisions that should be reviewed by you before the next implementation pass.

## 1) Notebook relocation finalization

- Current state:
  - Original notebook remains at `drowsiness_detection.ipynb` for backward compatibility.
  - A roadmap-aligned copy now exists at `notebooks/drowsiness_detection.ipynb`.
- Review needed:
  - Confirm whether I should fully move to `notebooks/` only (and remove root copy) in the next pass.

## 2) Experiment tracking backend upgrade

- Current state:
  - Phase 3 is implemented with local JSONL tracking at `artifacts/experiments/experiments.jsonl`.
- Review needed:
  - Confirm if you want me to implement MLflow integration now as Phase 3 extension.

## 3) Production artifact priority

- Current state:
  - Training/evaluation/inference use Keras `.keras` artifacts.
- Review needed:
  - Confirm preferred next artifact target:
    - TFLite export first, or
    - ONNX export first, or
    - both in one pass.

## 4) CI strictness level

- Current state:
  - CI uses practical smoke checks (`compileall`, synthetic training smoke, CLI checks).
- Review needed:
  - Confirm if you want heavier CI (dataset-backed train/eval jobs) despite longer runtime.

## 5) KPI policy approval

- Current state:
  - Business threshold is selected by `maximize(drowsy_recall - false_alarm_rate)`.
  - Current selected threshold from smoke model: `0.55`.
- Review needed:
  - Confirm if this objective is acceptable or if you want a different operating policy
    (for example: hard minimum drowsy recall target).

