# Advanced Getting Started Tutorial (MLOps)  
**Project:** Drowsiness Detection CV  
**Scope note:** `_Backup_CV/` is backup-only and out of active workflow.

---

## 1) What this tutorial gives you

By the end, you will be able to:

- run reproducible training/evaluation experiments,
- serve the model via API,
- generate monitoring and business KPI artifacts,
- run a production-style real-time simulation benchmark,
- operate the full project as an MLOps simulation pipeline.

---

## 2) Prerequisites

- Python 3.10+ (3.11 preferred long-term)
- Linux/macOS terminal (commands below are POSIX shell style)
- `pip` available
- Optional webcam/video file for real-time simulation

From repo root:

```bash
cd /path/to/Drowsiness-Detection-CV
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

> TensorFlow may print CUDA warnings on CPU-only machines. This is expected unless GPU stack is configured.

---

## 3) Understand the active architecture

Core paths:

- Config: `configs/config.yaml`
- Data: `Drowsy_dataset/train`, `Drowsy_dataset/test`
- Pipeline:
  - `src/train.py`
  - `src/evaluate.py`
  - `src/infer.py`
  - `src/api.py`
  - `src/realtime_webcam.py`
  - `src/monitoring.py`
  - `src/business_metrics.py`
  - `src/simulate_realtime.py`
- Outputs:
  - `artifacts/models/`
  - `artifacts/metrics/`
  - `artifacts/experiments/experiments.jsonl`
  - `artifacts/monitoring/inference_logs.jsonl`

Open the config and review defaults:

```bash
sed -n '1,220p' configs/config.yaml
```

---

## 4) First health check (fast)

Run synthetic smoke training to validate dependencies and graph compilation:

```bash
python3 -m src.smoke --samples 16
```

Run Python compile check:

```bash
python3 -m compileall src
```

If both pass, your environment is ready.

---

## 5) Reproducible baseline training

### 5.1 Run a full baseline training

```bash
python3 -m src.train \
  --version-name v1_baseline \
  --seed 42 \
  --epochs 10
```

### 5.2 Run a quick experiment (for iteration)

```bash
python3 -m src.train \
  --version-name v1_iter_quick \
  --seed 42 \
  --epochs 1 \
  --max-train-steps 20 \
  --max-val-steps 5
```

### 5.3 Training outputs

Expect a versioned folder:

```text
artifacts/models/<version_name>/
  model_best.keras
  model_final.keras
  history.json
  class_indices.json
  training_metadata.json
```

---

## 6) Evaluation pipeline and metrics artifacts

Evaluate a specific version:

```bash
python3 -m src.evaluate --version-name v1_baseline
```

Evaluate with custom threshold:

```bash
python3 -m src.evaluate --version-name v1_baseline --threshold 0.55
```

Artifacts are written to:

```text
artifacts/metrics/<version_name>/
  metrics.json
  predictions.csv
  confusion_matrix.png
  roc_pr_curve.png
```

---

## 7) Inference workflows

### 7.1 Single image inference

```bash
python3 -m src.infer \
  --model-path artifacts/models/v1_baseline/model_best.keras \
  --image Drowsy_dataset/test/DROWSY/618.png
```

### 7.2 Batch directory inference

```bash
python3 -m src.infer \
  --model-path artifacts/models/v1_baseline/model_best.keras \
  --input-dir Drowsy_dataset/test/DROWSY \
  --output-file artifacts/metrics/v1_baseline/inference_batch_drowsy.json
```

Inference outputs include:

- predicted label,
- positive-class probability,
- threshold used.

---

## 8) API deployment (FastAPI)

### 8.1 Start API with a pinned model

```bash
MODEL_PATH=artifacts/models/v1_baseline/model_best.keras \
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

### 8.2 Check health

```bash
curl --silent http://127.0.0.1:8000/health
```

### 8.3 Predict from image file

```bash
curl --silent -X POST \
  -F "file=@Drowsy_dataset/test/DROWSY/618.png" \
  "http://127.0.0.1:8000/predict?threshold=0.5"
```

Each API prediction is logged to:

```text
artifacts/monitoring/inference_logs.jsonl
```

### 8.4 Local real-time webcam detection (local-first runtime)

For latency-sensitive production-like usage, run detection locally instead of routing every frame through API:

```bash
python3 -m src.realtime_webcam \
  --model-path artifacts/models/v1_baseline/model_best.keras \
  --source 0 \
  --display
```

Headless mode (no OpenCV window):

```bash
python3 -m src.realtime_webcam \
  --model-path artifacts/models/v1_baseline/model_best.keras \
  --source 0 \
  --no-display \
  --max-frames 300
```

Quick readiness check (model + source):

```bash
python3 -m src.realtime_webcam --source 0 --health-check
```

Session summary is written to:

```text
artifacts/monitoring/realtime_session_<timestamp>.json
```

---

## 9) Monitoring summary generation

Generate a compact monitoring snapshot:

```bash
python3 -m src.monitoring \
  --output-file artifacts/monitoring/summary_latest.json
```

Summary includes:

- prediction distribution,
- average/min/max positive-class probability,
- total prediction count.

---

## 10) Business KPI and threshold strategy

Generate threshold trade-off and KPI report from evaluation predictions:

```bash
python3 -m src.business_metrics \
  --predictions-csv artifacts/metrics/v1_baseline/predictions.csv \
  --output-dir artifacts/metrics/v1_baseline/business
```

Add latency context from simulation benchmark:

```bash
python3 -m src.business_metrics \
  --predictions-csv artifacts/metrics/v1_baseline/predictions.csv \
  --benchmark-json artifacts/metrics/v1_baseline/realtime_benchmark.json \
  --output-dir artifacts/metrics/v1_baseline/business
```

Outputs:

- `threshold_tradeoff.csv`
- `kpi_report.json`

---

## 11) Production simulation (real-time benchmark)

> `src/simulate_realtime.py` is intended for benchmark simulation and metrics capture.  
> Use `src/realtime_webcam.py` for production-style local real-time operation.

### 11.1 Webcam mode

```bash
python3 -m src.simulate_realtime \
  --model-path artifacts/models/v1_baseline/model_best.keras \
  --source 0 \
  --frame-stride 3 \
  --max-frames 300 \
  --output-json artifacts/metrics/v1_baseline/realtime_benchmark.json \
  --display
```

### 11.2 Video file mode

```bash
python3 -m src.simulate_realtime \
  --model-path artifacts/models/v1_baseline/model_best.keras \
  --source path/to/video.mp4 \
  --frame-stride 3 \
  --max-frames 300 \
  --output-json artifacts/metrics/v1_baseline/realtime_benchmark.json
```

Benchmark output includes:

- average latency,
- p95 latency,
- max latency,
- effective FPS.

---

## 12) Experiment tracking (JSONL-first)

All train/evaluate events are appended to:

```text
artifacts/experiments/experiments.jsonl
```

Quick inspect:

```bash
tail -n 20 artifacts/experiments/experiments.jsonl
```

Parse and compare with Python:

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/experiments/experiments.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
train_rows = [r for r in rows if r.get("event_type") == "training"]
for row in train_rows[-5:]:
    print(row["timestamp_utc"], row["version_name"], row["metrics"])
PY
```

---

## 13) CI workflow

GitHub Actions workflow is at:

```text
.github/workflows/ci.yml
```

It runs:

- dependency installation,
- `compileall`,
- synthetic smoke training,
- CLI help checks.

Recommended local mirror:

```bash
python3 -m compileall src
python3 -m src.smoke --samples 8
python3 -m src.train --help
python3 -m src.evaluate --help
python3 -m src.infer --help
```

---

## 14) Advanced operating patterns

### Pattern A: Daily iteration loop

1. Quick train (`--epochs 1`, capped steps).  
2. Evaluate and inspect confusion matrix + PR curve.  
3. Adjust threshold via `src.business_metrics`.  
4. Validate API prediction path.

### Pattern B: Candidate release loop

1. Full train with named version (`v2_*`).  
2. Full evaluation + KPI report + benchmark.  
3. Store all artifacts under same version folder.  
4. Serve pinned model via `MODEL_PATH`.  
5. Run monitoring summary after test traffic.

### Pattern C: Reproducibility-focused loop

1. Fix seed and keep config checkpointed.  
2. Use explicit `--version-name`.  
3. Save and compare experiment rows (`experiments.jsonl`).  
4. Keep threshold policy file from KPI outputs.

---

## 15) Notebook workflow (optional, complementary)

- Active notebook copy: `notebooks/drowsiness_detection.ipynb`
- Root notebook remains for compatibility.
- Recommended usage:
  - use scripts for production-like runs,
  - use notebook for EDA, visual interpretation, and reporting.

---

## 16) Troubleshooting

### API fails at startup: model not found

- Ensure at least one trained model exists under `artifacts/models/<version>/`.
- Set explicit model path:

```bash
MODEL_PATH=artifacts/models/v1_baseline/model_best.keras uvicorn src.api:app --reload
```

### `src.evaluate` raises class or shape errors

- Ensure model and dataset are from same binary-class setup.
- Confirm image size/config consistency (`48x48x3` by default).

### TensorFlow GPU warnings

- If CPU-only, warnings are expected and can be ignored.
- If GPU intended, install matching CUDA/cuDNN stack for your TensorFlow version.

### No monitoring summary data

- Call `/predict` at least once (or run inference logging path) before:

```bash
python3 -m src.monitoring
```

---

## 17) Recommended next upgrades

- Add MLflow backend (keeping JSONL as fallback).
- Add ONNX/TFLite export jobs.
- Add dataset/version manifest and drift checks.
- Add stricter CI with dataset-backed mini train/eval tests.
- Add dashboard layer for KPI + monitoring trends.

---

## 18) One-command quickstart sequence

Use this when onboarding a new machine:

```bash
python3 -m venv .venv && source .venv/bin/activate && \
python3 -m pip install --upgrade pip && \
python3 -m pip install -r requirements.txt && \
python3 -m src.smoke --samples 8 && \
python3 -m src.train --version-name v1_baseline --epochs 1 --max-train-steps 20 --max-val-steps 5 && \
python3 -m src.evaluate --version-name v1_baseline && \
python3 -m src.infer --model-path artifacts/models/v1_baseline/model_best.keras --image Drowsy_dataset/test/DROWSY/618.png
```

This validates the full skeleton quickly before deeper training runs.
