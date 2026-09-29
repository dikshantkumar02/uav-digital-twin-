# AI-Enabled Real-Time Digital Twin for Uav Piston Engines (MALE UAV)

> **Research prototype** — not a specific certified or classified engine.
> The engine model in `config/engine.yaml` is a **representative** 4-cyl
> horizontally-opposed aero piston (~180 hp class) with **synthetic** /
> **public-data**-based defaults. Replace the maps with validated data
> before any operational use.

The system is being built in 16 phases (see [PHASE_STATUS.md](docs/PHASE_STATUS.md)).
This commit covers **PHASE 16**: testing + performance (coverage baseline,
pytest-benchmark, hypothesis invariants, fault-detection contract, stress).

## Layout

```
.
├── backend/
│   ├── config/        # YAML loader, Pydantic schemas, CLI   (PHASE 1)
│   ├── api/           # FastAPI endpoints                   (later)
│   ├── digital_twin/  # Physics + state estimator           (PHASE 6)
│   ├── environment/   # Atmosphere + disturbances           (PHASE 2)
│   ├── simulation/    # Mission + UAV + engine simulator    (PHASES 2-3)
│   ├── sensors/       # Virtual sensor model                (PHASE 4)
│   ├── telemetry/     # Streaming bus                       (PHASE 5)
│   ├── diagnostics/   # Anomaly / fault classifier          (PHASES 8-9)
│   ├── ml/            # ML pipelines                        (PHASES 8-9)
│   ├── health/        # Health index (per-subsystem)        (PHASE 10)
│   ├── rul/           # RUL + uncertainty                   (PHASE 11)
│   ├── risk/          # Mission risk engine                 (PHASE 12)
│   ├── dashboard/     # Real-time dashboard (FastAPI + WS)  (PHASE 13)
│   ├── hardware/      # Serial / UART transport + protocol  (PHASE 14)
│   ├── hil/           # HIL validation harness + golden compare (PHASE 15)
│   ├── testing/       # Invariant checkers + fault-detection contract + coverage helpers (PHASE 16)
│   ├── database/      # Persistence                         (later)
│   └── tests/         # Pytest suite
├── frontend/          # Dashboard (single self-contained HTML) (PHASE 13)
├── data/              # raw / processed / synthetic / scenarios
├── models/            # serialised ML artefacts
├── config/            # engine.yaml, environment.yaml, sensors.yaml, faults.yaml, risk.yaml, dashboard.yaml, hardware.yaml
├── scripts/           # operational scripts
├── tests/             # cross-cutting tests
├── docs/              # design notes, phase status, API docs
├── pyproject.toml,
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1. Create a virtual environment (Python 3.11+)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install PHASE 1 dependencies (only what is needed to validate config)
python -m pip install -U pip
python -m pip install pyyaml pydantic pytest

# 3. Validate the configuration
python -m backend.config.cli --config config --validate-only

# 4. Run PHASE 1 tests
python -m pytest backend/tests/test_phase1_config.py -v

# 5. (PHASE 13) Boot the real-time dashboard
python -m pip install fastapi 'uvicorn[standard]' 'websockets>=12' httpx
python -m backend.dashboard
# -> open http://localhost:8000/

# 6. (PHASE 14) The hardware transport is a library. The dashboard
# above already exercises the same TelemetryQueue via the synthetic
# StreamSource. To drive a real serial port, see backend/hardware/.
# Default config uses the in-process loopback — set
# config/hardware.yaml `port:` to /dev/ttyUSB0 (Linux) or
# /dev/cu.usbserial-* (macOS) to talk to a real device.

# 7. (PHASE 15) HIL validation harness. Records wire.bin +
#    snapshots.jsonl + manifest.yaml for each scenario, then replays
#    through the same LoopbackPort → SerialSource → TelemetryQueue
#    chain that live hardware would use, and compares each captured
#    DashboardSnapshot against the committed golden reference.
python -m backend.hil validate-all          # run the full validation
python scripts/generate_golden.py           # regenerate the golden set
```

The full `requirements.txt` lists every dependency across all phases; install it
inside the venv once those phases land:

```bash
python -m pip install -r requirements.txt
```

## Configuration provenance

Every numeric value in `config/*.yaml` is labelled:

| Label              | Meaning                                                 |
|--------------------|---------------------------------------------------------|
| `MEASURED`         | Taken from an instrumented test (must be supplied)      |
| `PUBLIC_DATA`      | Open literature / textbooks                             |
| `PHYSICS_DERIVED`  | Derived from first principles                           |
| `INTERPOLATED`     | Interpolated from public / measured points              |
| `SYNTHETIC`        | Synthesised for the prototype — not from a real engine  |
| `USER_CONFIGURED`  | Supplied by the operator at runtime                     |

## Engineering constraints

* **No deep learning for appearance.** Interpretable baselines first
  (Isolation Forest, Random Forest, Gradient Boosting).
* **No single-sensor thresholds** for fault decisions. Engine telemetry,
  flight state, environment, sensor health and Digital Twin residuals
  are all fused.
* **RUL is never reported as exact.** Returns an estimate with bounds,
  confidence, trend and an explicit `RUL_UNCERTAIN` status when
  evidence is insufficient.
* **No fake outputs.** Models that cannot produce a valid result return
  `INSUFFICIENT_DATA` / `MODEL_NOT_CALIBRATED` / `SENSOR_DATA_INVALID`.

## Phase progress

| Phase | Status   | Notes                                                   |
|-------|----------|---------------------------------------------------------|
| 1     | ✅ Done  | Project skeleton + configuration (15 tests)            |
| 2     | ✅ Done  | ISA1976 atmosphere, Dryden turbulence, gusts, mission (26 tests) |
| 3     | ✅ Done  | Representative piston engine model (26 tests)          |
| 4     | ✅ Done  | Virtual sensor system (22 tests)                       |
| 5     | ✅ Done  | Telemetry streaming (queue, preprocessor, latency) (19 tests) |
| 6     | ✅ Done  | Digital twin (predicted state + per-channel residual) (29 tests) |
| 7     | ✅ Done  | Fault injection (declarative scenarios + injector) (34 tests) |
| 8     | ✅ Done  | Anomaly detection (rule-based fusion of twin + sensor health) (32 tests) |
| 9     | ✅ Done  | Fault classification (Random Forest / Gradient Boosting) (22 tests) |
| 10    | ✅ Done  | Health index (5 subsystems + overall + trend) (38 tests) |
| 11    | ✅ Done  | RUL + uncertainty (closed form + RF, vectorised MC) (32 tests) |
| 12    | ✅ Done  | Mission risk engine (deficit-weighted sum, conservative status) (41 tests) |
| 13    | ✅ Done  | Real-time dashboard (FastAPI + WebSocket + single self-contained HTML) (29 tests) |
| 14    | ✅ Done  | Hardware interface (serial / UART transport + AERO wire protocol + loopback) (26 tests) |
| 15    | ✅ Done  | HIL validation harness (record / replay / golden compare) (23 tests) |
| 16    | ✅ Done  | Testing + performance: coverage baseline + perf budgets + benchmarks + property invariants + fault contract + 10× stress (26 tests) |
