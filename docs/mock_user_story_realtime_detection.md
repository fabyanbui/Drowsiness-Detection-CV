# Mock User Story: Real-Time Local Webcam Drowsiness Detection

## Context (as-is from current repository)

- `src/api.py` serves image-based inference through `POST /predict` (web/API flow).
- `src/simulate_realtime.py` supports webcam/video simulation and benchmark output.
- Current testing experience is still API/simulation oriented; a production-style real-time webcam feature is not yet packaged as a clear user-facing flow.

This mock story defines the product need for a local, low-latency real-time experience.

## Epic

**EPIC-RT-001: Local real-time drowsiness detection for production use**

Enable continuous drowsiness detection from a laptop webcam with local model execution to reduce network latency and improve alert responsiveness.

## Primary User Story

**US-RT-001**

As a driver safety user running the system on my laptop,  
I want the application to read my webcam feed and detect drowsiness in real time locally,  
so that I receive immediate alerts without depending on remote API latency.

## Supporting User Stories

**US-RT-002 (QA/Tester)**  
As a QA tester using Ubuntu on Windows 11 with an attached laptop camera, I want a repeatable local test flow for webcam detection so I can validate feature behavior before production rollout.

**US-RT-003 (Operations)**  
As an operations engineer, I want the model to run in local-first mode and expose health/status locally so I can monitor readiness without internet dependency.

**US-RT-004 (Product Owner)**  
As a product owner, I want real-time alerting capability available in the product workflow so the solution is practical for real production scenarios, not only offline evaluation.

## Acceptance Criteria (for US-RT-001)

1. The system can start in local mode and load a configured model artifact from local storage.
2. The system can access webcam source `0` by default and support camera source override.
3. The system performs continuous frame capture and real-time inference until the user exits.
4. The UI/console output shows current detection state (`DROWSY` or `NATURAL`) and confidence.
5. A drowsiness alert is triggered when drowsiness criteria are met (threshold and persistence rule).
6. Inference events are logged locally with timestamp, probability, predicted label, and latency.
7. If camera access fails, the system surfaces a clear error and does not continue silently.
8. The feature runs in Ubuntu on Windows 11 environment with laptop camera for manual validation.

## Business Value (mock)

- Reduces alert delay risk caused by remote API round-trip latency.
- Increases production readiness by enabling direct real-time usage.
- Improves operational reliability in low-connectivity or offline conditions.

