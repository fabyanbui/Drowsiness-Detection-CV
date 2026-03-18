# Business Requirements Document (BRD)
# Feature: Local Real-Time Webcam Drowsiness Detection

## 1) Document purpose

Define business-facing functional requirements for delivering a production-usable real-time drowsiness detection feature in this repository, with **local-first model serving** to reduce latency.

## 2) Business problem and opportunity

Current implementation is strong for script/API flows, but real production value requires continuous webcam detection with immediate alerts.  
If inference depends on exposed remote API only, round-trip latency can weaken real-time safety response.

## 3) Current-state assessment (from codebase)

- API capability exists: `src/api.py` provides `GET /health` and `POST /predict`.
- Real-time simulation exists: `src/simulate_realtime.py` supports webcam/video loop and outputs benchmark JSON.
- Gap: no formalized end-user real-time feature flow and acceptance baseline for production operations.

## 4) Target-state vision

Deliver a local runtime mode where webcam frames are analyzed continuously, drowsiness alerts are generated in near real time, and operational logs/health are available for production support.

## 5) Scope

### In scope

- Local deployment mode for real-time inference.
- Webcam-based continuous detection flow.
- Real-time detection state output and alerting behavior.
- Local logging, health checks, and testability in Ubuntu on Windows 11 setup.

### Out of scope (this BRD iteration)

- Model retraining strategy redesign.
- Cloud-scale orchestration and multi-node serving.
- Mobile/embedded deployment packaging.

## 6) Stakeholders

- Product Owner
- Safety Operations / End Users
- ML Engineer / MLOps Engineer
- QA Engineer

## 7) Functional requirements

| ID | Functional Requirement | Priority | Acceptance Criteria |
|---|---|---|---|
| FR-01 | The system shall support **local-first inference mode** where the model is loaded from local artifact path. | Must | Startup succeeds without internet dependency when model path is valid. |
| FR-02 | The system shall initialize webcam source with default index `0` and allow source override. | Must | User can run default camera; optional source argument switches camera/video input. |
| FR-03 | The system shall execute a continuous frame-capture loop for real-time inference. | Must | Frames are consumed continuously until manual stop or runtime error. |
| FR-04 | The system shall preprocess each inference frame with the same model input shape and normalization rules used in inference scripts. | Must | Real-time predictions are produced without shape/runtime mismatch errors. |
| FR-05 | The system shall classify each inference step into `DROWSY` or `NATURAL` using configurable threshold. | Must | Threshold parameter changes classification behavior as configured. |
| FR-06 | The system shall present current detection result to user in real time (label + confidence). | Must | Output updates during runtime and reflects most recent inference result. |
| FR-07 | The system shall trigger a drowsiness alert when configured drowsiness condition is met (threshold + persistence rule). | Must | Alert is raised when rule is met and clears when condition no longer met. |
| FR-08 | The system shall log inference events locally with timestamp, label, confidence, and latency fields. | Must | Log file contains structured entries for real-time session events. |
| FR-09 | The system shall expose local runtime health status (model loaded, camera status, runtime status). | Should | Health check command/endpoint returns operational readiness state. |
| FR-10 | The system shall provide configurable runtime parameters (threshold, frame stride, max frames/session mode). | Must | Parameters can be adjusted without code changes (CLI/config). |
| FR-11 | The system shall fail fast with explicit user-facing errors when model path or camera source is invalid. | Must | Invalid startup conditions produce clear errors and non-zero exit status. |
| FR-12 | The system shall support manual validation in **Ubuntu on Windows 11** environment using built-in laptop camera. | Must | QA checklist can be executed end-to-end on stated environment. |
| FR-13 | The system shall keep real-time operation available for production usage even when remote API is unavailable. | Must | Real-time local mode continues to operate during API/network outage. |
| FR-14 | The system shall preserve optional API compatibility for web consumers while prioritizing local real-time path for latency-sensitive usage. | Should | Existing API flow remains usable; local path is documented as preferred for real-time. |

## 8) Functional business rules

- BR-01: Real-time mode is **local by default** for latency-sensitive production operation.
- BR-02: Alert logic must be deterministic and configurable (threshold and persistence window).
- BR-03: Runtime failures (camera/model/config) must be surfaced explicitly; no silent fallback.
- BR-04: Real-time session logs must be written locally for audit and performance review.

## 9) Assumptions and dependencies

- Python environment with required packages from `requirements.txt`.
- Camera access is available from Ubuntu on Windows 11 setup.
- A trained model artifact is present under configured model directory.

## 10) Functional acceptance test scenarios (mock)

1. Start in local mode with valid model and webcam source `0`; verify continuous predictions and live state output.
2. Change threshold and verify alert behavior changes accordingly.
3. Disconnect/lock camera source and verify explicit startup/runtime error.
4. Run without internet and verify local real-time detection still operates.
5. Validate local log entries include required fields for each inference event.

## 11) Success criteria (functional)

- Real-time webcam detection can be executed locally end-to-end by QA.
- Drowsiness alerts are generated according to configured rules.
- Local health and logs support operational troubleshooting.
- Web/API flow remains available, but local mode is the recommended production path for real-time use.

