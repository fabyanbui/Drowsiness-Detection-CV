# Drowsiness Detection CV

> Dataset: [Link to the Drowsy Detection Dataset](https://www.kaggle.com/datasets/yasharjebraeily/drowsy-detection-dataset/data)

## MLOps roadmap implementation

This project is evolving from a notebook prototype into a modular MLOps simulation stack.

Active implementation now includes:

- Reproducible training pipeline (`src/train.py`)
- Evaluation pipeline with metrics + artifacts (`src/evaluate.py`)
- Inference pipeline (`src/infer.py`)
- API endpoint (`src/api.py`)
- Monitoring/business KPI helpers (`src/monitoring.py`, `src/business_metrics.py`)
- Real-time simulation benchmark (`src/simulate_realtime.py`)
- Local real-time webcam runtime (`src/realtime_webcam.py`)

> `_Backup_CV/` is intentionally out of active scope.

## Project structure

```text
.
├── Drowsy_dataset/
├── artifacts/
├── configs/
│   └── config.yaml
├── docs/
├── notebooks/
│   └── drowsiness_detection.ipynb
├── src/
│   ├── api.py
│   ├── business_metrics.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── modeling.py
│   ├── monitoring.py
│   ├── realtime_webcam.py
│   ├── simulate_realtime.py
│   ├── smoke.py
│   ├── tracking.py
│   ├── train.py
│   ├── utils.py
│   └── versioning.py
└── requirements.txt
```

## Quick start

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Train a baseline model:

```bash
python3 -m src.train
```

Evaluate the latest model:

```bash
python3 -m src.evaluate
```

Run inference on one image:

```bash
python3 -m src.infer --image path/to/image.png
```

Start API:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Run local real-time webcam detection (recommended for latency-sensitive production usage):

```bash
python3 -m src.realtime_webcam --source 0 --display
```

Health-check local real-time dependencies (model + capture source):

```bash
python3 -m src.realtime_webcam --source 0 --health-check
```

Run synthetic smoke test:

```bash
python3 -m src.smoke
```
