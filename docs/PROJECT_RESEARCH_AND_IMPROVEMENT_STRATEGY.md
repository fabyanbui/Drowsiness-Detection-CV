# Drowsiness-Detection-CV: Project Research & Improvement Strategy

## Scope and review method

I reviewed the active project files in this repository and intentionally **excluded `_Backup_CV/`** as requested.

Reviewed assets:

- `README.md`
- `requirements.txt`
- `drowsiness_detection.ipynb`
- `Drowsy_dataset/` (train/test folders and class distributions)

Important finding: there are **no active `.py` files** in the current project scope outside backup; the main logic is notebook-driven.

## Current project snapshot

### Data assets

- Dataset structure:
  - `Drowsy_dataset/train/{DROWSY,NATURAL}`
  - `Drowsy_dataset/test/{DROWSY,NATURAL}`
- File counts:
  - Train: `DROWSY=2809`, `NATURAL=3050` (total `5859`)
  - Test: `DROWSY=757`, `NATURAL=726` (total `1483`)
- Class balance is relatively healthy (roughly near 50/50 in each split).
- All sampled images are `.png` and consistently `48x48`.

### Modeling workflow (from notebook)

- Uses `ImageDataGenerator` with augmentation, rescaling, and `validation_split=0.2`.
- CNN architecture:
  - `Conv2D(32)` -> `MaxPool2D` -> `Flatten` -> `Dense(64)` -> `Dense(1, sigmoid)`
- Loss/optimizer:
  - `binary_crossentropy` + `adam`
- Training:
  - 10 epochs
- Testing:
  - Evaluates on `test_data_gen`
- Last recorded notebook outputs:
  - training accuracy ~`0.7188`
  - test accuracy ~`0.8577`

## Senior DS/BA assessment: strengths and gaps

### Strengths

- End-to-end prototype exists (load data -> train -> evaluate).
- Dataset split is already separated into train/test folders.
- Baseline CNN is lightweight and suitable for quick iteration.

### High-impact technical gaps

1. **Validation leakage risk in evaluation quality**
   - Validation generator is created from the same augmented generator used for training.
   - Best practice: augmentation for train only; validation/test should be rescale-only.

2. **Evaluation is too narrow for a safety-related use case**
   - Only accuracy/loss are tracked.
   - Missing precision, recall, F1, confusion matrix, ROC-AUC/PR-AUC, threshold analysis.

3. **Reproducibility is weak**
   - No fixed seeds for Python/NumPy/TensorFlow.
   - Test generator uses `shuffle=True`, which is not ideal for stable analysis/reporting.

4. **Notebook narrative and code are inconsistent**
   - Markdown says softmax/2-class output while code uses sigmoid/1-node output.
   - Markdown says `Dense(128)` while code uses `Dense(64)`.
   - Some chart labels/titles are mismatched (accuracy vs loss naming confusion).

5. **Prototype is not production-ready**
   - No model export/versioning strategy.
   - No inference pipeline for real-time stream logic.
   - No monitoring/drift plan.

6. **Business-readiness is underdeveloped**
   - No explicit product KPIs (false alarms/hour, missed drowsy events, alert latency).
   - No cost-based decision threshold strategy.
   - No stakeholder-facing dashboard/reporting layer.

## Improvement strategy aligned with Data Scientist + Business Analyst JD expectations

Below is a strategy designed to map to common JD requirements: problem framing, robust modeling, analytics communication, deployment readiness, and measurable business impact.

### Priority 1: Analytics and model reliability foundation

- Separate data generators clearly:
  - Train: augmentation + rescale
  - Validation/Test: rescale only
- Add deterministic settings:
  - Global seeds + deterministic ops where feasible
- Expand evaluation:
  - confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC
  - per-class metrics for `DROWSY` sensitivity
- Introduce callbacks:
  - `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`
- Save artifacts:
  - best model, training history, metrics JSON

### Priority 2: Stronger modeling and experimentation discipline

- Establish baselines:
  - simple CNN baseline (current), transfer-learning baseline (e.g., MobileNetV2)
- Build a repeatable experiment tracker:
  - log parameters, metrics, confusion matrices, model versions
- Tune threshold by business cost:
  - prioritize reducing **false negatives** (missed drowsiness) while controlling alert fatigue
- Add error analysis:
  - inspect false positives/false negatives by lighting, head pose, eye occlusion

### Priority 3: Product and business analytics layer

- Define operational KPIs for decision-making:
  - drowsy recall at operating threshold
  - false alarms per driving hour
  - mean time-to-alert from drowsy onset
  - model confidence calibration quality
- Build stakeholder dashboard:
  - weekly trend of KPIs, confusion matrix drift, threshold scenarios
- Create a business impact model:
  - expected incident-risk reduction vs alert burden trade-off

### Priority 4: Deployment and MLOps maturity

- Convert notebook workflow into modular code:
  - `src/data.py`, `src/train.py`, `src/evaluate.py`, `src/infer.py`
- Add CI checks:
  - data path validation, minimal training smoke test, evaluation script run
- Prepare edge deployment:
  - export to TFLite/ONNX, benchmark latency and memory
- Add monitoring plan:
  - input drift, class distribution drift, confidence drift, periodic re-training trigger

## JD competency mapping (practical)

- **Business problem translation** -> Define KPIs and threshold policy tied to safety and alert fatigue.
- **Data quality and feature rigor** -> Add systematic data audits and error slicing.
- **Model development** -> Baselines, transfer learning, rigorous metrics.
- **Experimentation and statistics** -> Controlled experiments, threshold optimization, reproducibility.
- **Communication/storytelling** -> Dashboard and decision-oriented reporting.
- **Production collaboration** -> Modular code, versioned artifacts, CI, deployment benchmarks.

## Recommended execution order (pragmatic)

1. Fix evaluation reliability (generator split, metrics, seeds, callbacks).
2. Create reproducible train/eval scripts from notebook logic.
3. Define business KPIs and threshold policy with stakeholders.
4. Run model benchmarking + error analysis to pick deployment candidate.
5. Package for edge inference and add monitoring/reporting.

---

If you want, the next step can be converting this notebook into a clean Python training pipeline with reproducible experiments and a KPI-oriented evaluation report.
