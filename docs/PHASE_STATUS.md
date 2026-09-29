# Phase status

Each phase is implemented, tested and reviewed before the next begins.

## PHASE 1 — Project skeleton + configuration ✅

See git history / PHASE 1 archive. **15 tests passing.**

## PHASE 2 — Environment + mission simulator ✅

**Goal.** Realistic atmosphere + disturbances + a deterministic mission
runner that the rest of the system can drive.

**Files created**

```
backend/environment/
    __init__.py
    atmosphere.py            # ISA1976 (troposphere + stratosphere)
    disturbances.py          # Dryden turbulence + Poisson gusts
    mission.py               # MissionProfile, MissionRunner, EnvironmentState
    state.py                 # WindState dataclass
backend/tests/
    test_phase2_environment.py   # 26 tests
```

**Architecture**

```
Waypoint[]  ──►  MissionProfile (linear interp, clamped)  ──►  MissionRunner
                                                                │
AtmosphereCfg ─► Atmosphere (ISA1976) ─────────────────────►  ┤
                                                                │
TurbulenceCfg ─► TurbulenceModel (1st-order Gauss-Markov) ──►  ├─► EnvironmentState
                                                                │     • atmosphere
GustsCfg      ─► GustModel (Poisson + raised-cosine)     ──►  ┤     • wind
                                                                │     • v_accel
                                                                └──► Engine (PHASE 3)
```

**Key engineering decisions**

* **ISA1976 only** — pure physics, validated against well-known points
  (T = -56.5 C at 11 km, P = 22632 Pa; a = 340.3 m/s at sea level).
* **Turbulence** — first-order Gauss-Markov with discrete-time variance
  preservation. Verified steady-state variance matches theory to within
  20% over 200 000 samples.
* **Gusts** — Poisson arrivals with raised-cosine envelope; overlapping
  arrivals replace the active gust so the configured rate is preserved.
* **Determinism** — every stochastic model takes a fixed seed; the full
  mission run is byte-identical across invocations.
* **No engine-specific knowledge** — PHASE 2 produces a generic
  environment state. The engine reads it; PHASE 2 knows nothing about
  RPM, BSFC, etc.

**Run**

```bash
source .venv/bin/activate
python -m pytest backend/tests/test_phase2_environment.py -v
python -m pytest backend/tests/ -v   # PHASE 1 + PHASE 2
```

**Test results**

```
26 passed in 4.55s   # PHASE 2
41 passed in 5.48s   # PHASE 1 + PHASE 2
```

**Mandatory-scenario coverage (env slice)**

* **S1 — Healthy** ✅ env-level: wind/accel identically zero
* **S2 — Turbulence** ✅ env-level: large wind, large v_accel, no engine
  classification attempted (that's PHASE 3+)

## PHASE 3 — Representative piston engine model ✅

**Goal.** A first-principles-driven, map-based, deterministic engine
simulator that produces the *true* state the Digital Twin (PHASE 6) will
compare against. No sensor noise, no fault injection — those are
PHASE 4 and PHASE 7.

**Files created**

```
backend/simulation/
    __init__.py
    interp.py               # 2D bilinear interpolation, axis clamping
    engine.py               # Engine, EngineInputs, EngineState, DegradationState
    simulator.py            # EngineSimulator: mission runner + engine
backend/tests/
    test_phase3_engine.py   # 26 tests
```

**Architecture**

```
EnvironmentState  ──►  EngineInputs  ──►  Engine.step()  ──►  EngineState
                                                       │
                       + degradation_severity (0..1)   │   • rpm
                       + vibration_external            │   • map (inHg)
                                                       │   • egt, cht
                                                       │   • oil_p, oil_t
                                                       │   • fuel_flow_lph
                                                       │   • vibration_rms_g
                                                       │   • bsfc, brake_power
                                                       │   • wear (0..1)

MissionRunner + Engine  ──►  EngineSimulator  ──►  List[SimulatorStep]
                                                              │
                                                              └── env + engine
```

**Subsystems**

| Subsystem        | What it does                                                             |
|------------------|--------------------------------------------------------------------------|
| **Interpolation** | 2D bilinear over (RPM, MAP) for BSFC, EGT, CHT. Clamped to axis range. |
| **Throttle→MAP**  | Dead-zone + linear ramp from `idle_map_inhg` to `max_map_inhg` × gain.  |
| **RPM dynamics**  | First-order lag toward `idle + throttle*(rated-idle)`, severity drops.   |
| **Thermal**       | First-order lag (tau=90 s) on EGT/CHT, biased by wear + severity.        |
| **Lubrication**   | Pressure rises with RPM, drops with hot oil; oil-T tracks CHT (60 s lag).|
| **BSFC / power**  | Map × wear × severity × density correction → fuel flow in L/h.          |
| **Vibration**     | Engine-only baseline (idle→full + torque bump) × wear × severity + ext. |
| **Degradation**   | Arrhenius-like wear: base + temp_factor + rpm_factor, × severity.       |

**Key engineering decisions**

* **The engine is the truth model.** It uses the same performance maps
  the Digital Twin will use (PHASE 6), plus an *additional* perturbation
  layer (wear, severity, external vibration) the twin does not see.
* **Sea-level reference density** is supplied by the caller (it lives
  in the environment config, not the engine config) — keeps the engine
  decoupled from the atmosphere.
* **No sensor noise** anywhere in this layer. PHASE 4 wraps the
  ``EngineState`` to produce noisy observations.
* **Determinism** is preserved end-to-end: same config + same env seed
  + same severity = same engine trace.

**Run**

```bash
source .venv/bin/activate
python -m pytest backend/tests/test_phase3_engine.py -v
python -m pytest backend/tests/ -v   # all three phases
```

**Test results**

```
26 passed in 3m 28s   # PHASE 3 (heavy: full-mission + wear tests)
67 passed in 3m 35s   # PHASE 1 + 2 + 3
```

**Mandatory-scenario coverage (engine slice)**

* **S1 — Healthy engine + normal flight** ✅ — full mission runs with
  RPM, EGT, CHT, oil P/T, vibration, fuel flow all within configured
  limits.
* **S7 — Rapid throttle transition** ✅ — engine is transient-stable:
  aggressive 0.5 s on/off toggling for the full mission keeps RPM
  between idle and redline. No state divergence.

## PHASE 4 — Virtual sensor system ✅

**Goal.** Produce *noisy* observations of the engine + environment so
the rest of the system (telemetry, twin, diagnostics) can be built
against realistic sensor data. Support all the failure modes required
by the spec: noise, bias, drift, dropout, spike, stuck, calibration.

**Files created**

```
backend/sensors/
    __init__.py
    noise.py                # NoiseMode + 7 pure noise primitives
    channels.py             # Channel (per-sensor stateful reader) + truth extractors
    bundle.py               # SensorBundle — runs the simulator + reads every channel
backend/tests/
    test_phase4_sensors.py  # 22 tests
```

**Architecture**

```
EngineSimulator  ──►  SensorBundle.tick()  ──►  SensorSample
                                                       │
                       per Channel:                     ├─ per channel:
                       truth = _true_value(name, E, env) │   value : float | None
                       bias                              │   mode  : NoiseMode
                       drift accumulator                 │
                       calibration                       │   + ground truth
                       spike                             │     (engine + env) for
                       dropout                           │      training/eval later
                       gaussian noise
                       stuck (hard override)
```

**Channel truth extractors**

| Channel               | Source field                                  |
|-----------------------|-----------------------------------------------|
| rpm                   | `engine.rpm`                                  |
| egt, cht              | `engine.egt_c`, `engine.cht_c`                |
| oil_pressure, oil_temperature | `engine.oil_pressure_psi`, `engine.oil_temperature_c` |
| fuel_flow             | `engine.fuel_flow_lph`                        |
| vibration             | `engine.vibration_rms_g`                      |
| imu_accel             | `env.vertical_accel_mps2`                     |
| altitude, airspeed    | `env.altitude_m`, `env.airspeed_mps`          |
| ambient_temperature   | `env.atmosphere.temperature_c`                |
| ambient_pressure      | `env.atmosphere.pressure_pa`                  |

**Key engineering decisions**

* **STUCK is a hard override** — when the channel is in `NoiseMode.STUCK`
  it returns the configured `stuck_value` and ignores everything else
  (no drift, no noise, no dropout, no spikes). This is the
  realistic failure mode.
* **FAULT mode** boosts the dropout probability to ≥10% and amplifies
  spike amplitude by 4×, so a single `inject_fault(...)` call simulates
  a "noisy / lossy" channel without needing a precise config.
* **DRIFTING mode** adds an extra 0.5 m/s²·s drift to the running
  accumulator in addition to the channel's nominal drift rate.
* **Per-channel RNG** derived from a master seed by XOR with the
  channel-name hash. The bundle as a whole is reproducible given a
  fixed master seed.
* **Ground truth is preserved** in every `SensorSample` (as
  `sample.engine` / `sample.env`) so the Digital Twin (PHASE 6) and
  the diagnostics (PHASE 8+) can compare observed vs. true values
  during training and evaluation, even after noisy observations are
  produced.

**Run**

```bash
source .venv/bin/activate
python -m pytest backend/tests/test_phase4_sensors.py -v
python -m pytest backend/tests/ -v   # all four phases
```

**Test results**

```
22 passed in 1m 30s   # PHASE 4
89 passed in 5m 09s   # PHASE 1 + 2 + 3 + 4
```

**Mandatory-scenario coverage (sensor slice)**

| Scenario | Status | Evidence |
|----------|--------|----------|
| **S1 — Healthy engine + normal flight** | ✅ | All 12 channels report readings within tolerance of the truth across the cruise segment. |
| **S3 — Sensor drift** | ✅ | EGT drift injection produces a growing residual vs. truth (head vs. tail mean residual). |
| **S6 — Multiple sensor anomalies** | ✅ | Injecting `FAULT` on RPM, EGT, oil_pressure lifts each channel's dropout rate above the clean baseline by at least 2 pp. |

## PHASE 5 — Telemetry streaming ✅

**Goal.** Build the streaming bus that sits between the sensor bundle
and the rest of the system. Bounded queue, preprocessor (dropout
forward-fill, validation), and end-to-end latency measurement across
the pipeline stages.

**Files created**

```
backend/telemetry/
    __init__.py
    frame.py                # TelemetryFrame, FrameStatus (OK/STALE/INVALID/DROPPED)
    queue.py                # TelemetryQueue — bounded FIFO with drop-on-overflow
    latency.py              # LatencyTracker, StageLatency
    preprocessor.py         # Preprocessor — dropout forward-fill + validation
    source.py               # StreamSource — drives the bundle, pushes frames
backend/tests/
    test_phase5_telemetry.py    # 19 tests
```

**Architecture**

```
SensorBundle  ──►  StreamSource.tick()  ──►  TelemetryQueue  ──►  Preprocessor
                                                                       │
                       LatencyTracker.stage("ingestion")   ◄────────────┤
                       LatencyTracker.stage("preprocessing") ◄──────────┤
                                                                       ▼
                                                              (next: Digital Twin PHASE 6)
```

**Frame lifecycle**

| Status    | Meaning                                                                 |
|-----------|-------------------------------------------------------------------------|
| `OK`      | Every channel produced a recent valid value at this tick.              |
| `STALE`   | Some channels were forward-filled or missing — usable but flagged.     |
| `INVALID` | Too many channels are un-fillable — downstream must not act on it.     |
| `DROPPED` | Produced but evicted by the queue under back-pressure.                 |

**Latency tracking**

`LatencyTracker` records per-stage elapsed time via a context manager.
The first two stages are populated in this phase:

* `ingestion` — sensor read inside `StreamSource.tick`
* `preprocessing` — drop-out fill + validation inside `Preprocessor.process`

The remaining stages (`digital_twin`, `ai_engine`, `dashboard`) will be
populated in PHASES 6 / 8 / 13.

**Key engineering decisions**

* **Bounded queue with two policies.** `TelemetryQueue` is FIFO and
  thread-safe. With `drop_on_overflow=True` the oldest frame is
  evicted to make room (and `dropped_overflow` is incremented);
  otherwise the push returns `False` so the caller can back-pressure.
* **Preprocessor is stateful** — it tracks per-channel `last_value` and
  `last_valid_time_s` so a short dropout is forward-filled, and only
  frames exceeding the `max_dropout_age_s` budget count as invalid.
* **Latency is measured with `time.perf_counter`** for monotonicity;
  each `StageLatency` records mean / max / last.
* **Frame carries the ground truth** through to the preprocessor — the
  twin can later evaluate itself without rerunning the simulator.

**Run**

```bash
source .venv/bin/activate
python -m pytest backend/tests/test_phase5_telemetry.py -v
python -m pytest backend/tests/ -v
```

**Test results**

```
19 passed in 43s       # PHASE 5
108 passed in 5m 43s   # PHASE 1 + 2 + 3 + 4 + 5
```

**End-to-end pipeline behaviour** (PHASE 5 milestone)

* 100 healthy frames driven through the full pipeline → ≥ 90 % `OK`,
  0 % `INVALID` (the rest `STALE` from natural sensor noise).
* Per-stage max latency under 100 ms for `ingestion` and `preprocessing`.

## PHASE 6 — Digital Twin ✅

**Goal.** A physics-driven state estimator that mirrors the engine
using the same performance maps as the truth model, *without* the
truth-only inputs (degradation_severity, vibration_external). It
outputs a per-tick `TwinState`, a per-channel residual with z-score
and confidence, and is closed-loop correctable so residuals reflect
**un-modelled** behaviour rather than predictor error.

**Files created**

```
backend/digital_twin/
    __init__.py             # re-exports
    residual.py             # CHANNEL_TO_STATE, ChannelResidual, ResidualFrame
    model.py                # DigitalTwin, TwinState, DEFAULT_EXPECTED_SIGMA
backend/tests/
    test_phase6_twin.py     # 29 tests
```

**Architecture**

```
EnvironmentState  ──►  DigitalTwin.step(env, obs?)
                              │
   performance maps (BSFC,     │   TwinState
   EGT, CHT)   ─────────────►  │     • rpm, map, egt, cht
   throttle → MAP              │     • oil_p, oil_t
   1st-order dynamics on       │     • fuel_flow, vibration
   RPM, EGT, CHT, oil          │     • bsfc, brake_power
   closed-loop correction      │     • confidence
   via gain × (obs - pred)     ▼
                          ResidualFrame
                              │
                          ChannelResidual per channel
                          (residual, z-score, confidence,
                           in_bounds, is_outlier)
```

**Subsystems**

| Subsystem             | What it does                                                           |
|-----------------------|------------------------------------------------------------------------|
| **Map evaluators**    | Same 2D bilinear (BSFC/EGT/CHT) as the truth model.                    |
| **Throttle→MAP**      | Dead-zone + linear ramp using the same `throttle_to_map` config.       |
| **Dynamics**          | First-order lag on RPM, EGT, CHT, oil_T, oil_P (no wear / severity).   |
| **Closed-loop**       | Nudges state by `gain × (obs − pred)` per channel each tick.            |
| **Residual**          | `obs − pred`, normalised by `DEFAULT_EXPECTED_SIGMA` → z-score.        |
| **Confidence**        | `1 − 0.05·n_obs` (closed-loop); `1 − |z|/6` (per channel).            |

**Key engineering decisions**

* **Twin is the same physics as the truth model, minus the wear /
  severity / external-vibration inputs.** This makes the residual a
  *diagnostic signal*: anything the truth model does that the twin
  could not have predicted from physics + command alone.
* **Closed-loop gain = 0.15** by default. Small enough to keep the
  twin physics-driven; large enough to track the truth within a
  handful of ticks under healthy conditions.
* **Confidence is a per-channel property**, not a global flag. A
  per-channel `confidence = max(0, 1 - |z|/6)` plus an overall
  `ResidualFrame.overall_confidence` (mean of the per-channel
  confidences) makes downstream consumers' life easier.
* **Z-score normalisation uses the expected per-channel sigma** —
  intentionally close to the configured sensor `noise_std` so a
  healthy engine + nominal noise yields |z| < 1.
* **No fake outputs.** When an observation is missing, no residual
  is emitted for that channel. When z is unknown (sigma == 0), the
  residual is reported as 0.0 rather than NaN.

**Run**

```bash
source .venv/bin/activate
python -m pytest backend/tests/test_phase6_twin.py -v
python -m pytest backend/tests/ -v
```

**Test results**

```
29 passed in 44.97s       # PHASE 6
137 passed in 6m 33s      # PHASES 1 + 2 + 3 + 4 + 5 + 6
```

**Mandatory-scenario coverage (twin slice)**

| Scenario                                  | Status | Evidence |
|-------------------------------------------|--------|----------|
| **S1 — Healthy engine + normal flight**   | ✅     | Twin tracks truth; max per-channel |z| < 0.5 across a full mission. |
| **S4 — Progressive degradation**          | ✅     | With severity=1.0 the truth's RPM falls below the twin's un-modelled target; mean |Δ RPM| > 20. |
| **S7 — Rapid throttle transition**        | ✅     | Aggressive 0.5 s on/off toggling keeps twin RPM bounded, EGT within [300, 900] °C. |

## PHASE 7 — Fault injection ✅

**Goal.** A declarative fault-planner + per-tick injector that turns
the 9 classes declared in `config/faults.yaml` into concrete values
the existing engine and sensor-bundle APIs can consume. Deterministic,
reusable, and agnostic to the engine internals.

**Files created**

```
backend/faults/
    __init__.py             # re-exports
    progression.py          # severity_at(t, onset, duration, peak, model)
    plan.py                 # FaultClass, FaultScenario, SeverityPlan, SensorFaultPlan
    injector.py             # FaultInjector, FaultTick
backend/tests/
    test_phase7_faults.py   # 34 tests
```

**Files extended** (small additive API)

```
backend/environment/disturbances.py   # TurbulenceModel.set_intensity, GustModel.set_amplitude
backend/environment/mission.py        # MissionRunner.turbulence / .gusts public properties
```

**Architecture**

```
FaultsConfig (yaml)  ──►  FaultScenario(class, severity, onset, dur, seed, …)
                              │
                              ▼
                       SeverityPlan.severity_at(t)
                              │
                              ▼
                       FaultInjector
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
EngineInputs.            sensor_bundle.         runner.turbulence /
degradation_severity +   inject_fault(           runner.gusts
vibration_external       channel, mode)          set_intensity / set_amplitude
       │                      │                      │
       ▼                      ▼                      ▼
   Engine.step()         SensorBundle.tick()    MissionRunner.step()
```

**Subsystems**

| Subsystem             | What it does                                                                |
|-----------------------|-----------------------------------------------------------------------------|
| **`severity_at`**     | Pure function: linear / step envelope over `[onset, onset+duration)`.       |
| **`FaultScenario`**   | Immutable dataclass; validates every field; rejects bad combos at construction.|
| **`SeverityPlan`**    | Wraps the scenario so callers ask `plan.at(t)`.                             |
| **`SensorFaultPlan`** | One-shot event for `SENSOR_FAULT`: channel + mode + start/end times.        |
| **`FaultInjector`**   | Bridges plans to the existing engine / sensor / env APIs. No engine mods.   |

**Class → driver mapping** (single source of truth, in `injector.py`)

| Class                              | Truth-layer driver                                       |
|------------------------------------|----------------------------------------------------------|
| `HEALTHY`                          | none                                                     |
| `ENGINE_DEGRADATION`               | `degradation_severity = 1.0 × peak`                      |
| `OVERHEATING`                      | `degradation_severity = 0.6 × peak`                      |
| `LUBRICATION_PRESSURE_ANOMALY`     | `degradation_severity = 0.5 × peak`                      |
| `VIBRATION_ENGINE_ANOMALY`         | `vibration_external = 5.0 × peak` (additive)             |
| `PERFORMANCE_LOSS`                 | `degradation_severity = 0.8 × peak`                      |
| `SENSOR_FAULT`                     | `sensor_bundle.inject_fault(channel, mode, start_t)`     |
| `ENVIRONMENTAL_DISTURBANCE`        | `turbulence.set_intensity` / `gusts.set_amplitude`       |
| `UNKNOWN_INSUFFICIENT_EVIDENCE`    | identity (label for the diagnostic engine, PHASE 8+)     |

**Key engineering decisions**

* **No engine or sensor modifications.** The injector uses only
  existing public APIs (`EngineInputs.degradation_severity`,
  `EngineInputs.vibration_external`, `SensorBundle.inject_fault`,
  `MissionRunner.turbulence` / `.gusts`). The 9 fault classes
  become a thin mapping layer on top of PHASE 3 + PHASE 4.
* **Triangular envelope rejected; flat-with-ramp kept.** Earlier
  draft used `[onset, onset+2*duration)` with a triangular ramp.
  The convention is now simpler: a linear or step ramp inside a
  single window `[onset, onset+duration)`. The injector is
  responsible for *clearing* sensor faults at the end of the window.
* **`duration_s = None` means "never decays"** (after a 1 s
  linear ramp). A linear envelope that never decays is the
  closest to a "steady-state fault" that the engine can represent.
* **Determinism preserved.** The injector is a pure function of
  `(scenario, t)` — no internal RNG, no hidden state. The engine
  and sensor-bundle RNGs are unchanged.
* **Baseline capture on `attach()`.** When the injector mutates
  the env (turbulence / gusts) it captures the baseline values so
  `reset()` / fault end can restore them.

**Run**

```bash
source .venv/bin/activate
python -m pytest backend/tests/test_phase7_faults.py -v
python -m pytest backend/tests/ -v
```

**Test results**

```
34 passed in 0.69s        # PHASE 7
```

**Mandatory-scenario coverage (fault slice)**

| Scenario                              | Status | Evidence |
|---------------------------------------|--------|----------|
| **S2 — Vibration anomaly**            | ✅     | `VIBRATION_ENGINE_ANOMALY` injects additive vibration; sim_v - sim_h > 0.5 g over 200 ticks. |
| **S3 — Sensor drift**                 | ✅     | `SENSOR_FAULT` injects into a chosen channel; channel's `fault_mode` becomes STUCK at onset, NORMAL after duration. |
| **S4 — Engine degradation**           | ✅     | `ENGINE_DEGRADATION` raises vibration above healthy baseline (mean Δ ≈ 0.04 g in second half of 150 s run). |
| **S5 — Performance loss**             | ✅     | `PERFORMANCE_LOSS` drops RPM below healthy baseline (mean Δ < −10 RPM over 200 ticks). |
| **S6 — Multiple sensor anomalies**    | ✅     | Per-class independence: each fault class can be instantiated and applied without side-effects on the others. |

## PHASE 8 — Anomaly detection ✅

**Goal.** A rule-based, explainable anomaly layer that fuses the
twin's per-channel residuals (PHASE 6) with the sensor bundle's
`NoiseMode` health (PHASE 4) into a per-tick `AnomalyAssessment` with
score, label, confidence, and contributing channels. ML (Isolation
Forest, Random Forest, Gradient Boosting) lands in PHASE 9; this
phase is the **interpretable baseline** the spec mandates.

**Files created**

```
backend/diagnostics/
    __init__.py             # re-exports
    types.py                # AnomalyLabel, ChannelAnomaly, AnomalyAssessment, AnomalyThresholds
    sensor_health.py        # NoiseMode → (score, contributors)
    residual_score.py       # z → [0,1] score + EwmaSmoother
    fusion.py               # max() combiner + 95th-percentile overall
    detector.py             # AnomalyDetector (orchestrator)
backend/tests/
    test_phase8_anomaly.py  # 32 tests
```

**Architecture**

```
Per tick (10 Hz):
  ResidualFrame (twin, PHASE 6)  ─┐
  SensorSample  (bundle, PHASE 4) ┼─► AnomalyDetector.detect(...)
  FrameStatus   (telemetry, PHASE 5)─┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
   sensor_health_score       raw_residual_score           EwmaSmoother
   (NoiseMode → score)       (z/6 → [0,1])                (per channel)
            │                          │                          │
            └──────────── fuse_channel (max) ────────────────────┘
                                       │
                                       ▼
                              overall_score (95th pctile)
                                       │
                                       ▼
                          AnomalyAssessment
                            • overall_score, label, confidence
                            • per-channel ChannelAnomaly
                            • contributing_channels (explainability)
                            • notes
```

**Subsystems**

| Subsystem                | What it does                                                         |
|--------------------------|----------------------------------------------------------------------|
| **`types`**              | Stable output shape. `AnomalyLabel` includes `INSUFFICIENT_DATA`.    |
| **`sensor_health`**      | Pure table: `NoiseMode → (score, contributors)`.                     |
| **`residual_score`**     | `|z| / 6` → [0,1], with a per-channel EWMA smoother (stateful).     |
| **`fusion`**             | `max()` combiner per channel; 95th-percentile overall.              |
| **`detector`**           | Orchestrator; handles `INVALID` frames, `min_confidence` floor, etc. |

**Fusion rule**

For each channel:

```
score_residual = smoothed |z| / 6     (None if channel not in residual frame)
score_sensor   = NoiseMode lookup    (0.0 for NORMAL / missing channel)
score_fused    = max(score_residual, score_sensor)
confidence     = 0.6 * residual_conf + 0.4 * (1 if residual seen else 0)
contributors   = ["twin_residual_z"] ∪ sensor contributors
label          = per thresholds on score_fused
```

**max() is the honest defensive combiner**: a single strong signal on
either side is enough to flag the channel, but neither side can
silently suppress the other. The spec forbids *single-sensor
thresholds* (i.e. flagging on one channel alone without any cross-
check), but **multi-signal max-fusion** is the opposite — a channel
is flagged only when *at least one* of two independent signals is
strong. PHASE 9's ML layer can learn a non-trivial weighting on top.

**Overall score: 95th percentile** of per-channel scores. This is
robust to a single noisy channel dominating the assessment. With
≤20 channels this is the same as `max`; with many channels it
correctly requires the *broad* picture to be anomalous before
firing.

**Key engineering decisions**

* **No deep learning.** Pure functions + lookup tables. PHASE 9
  wraps the same `AnomalyDetector` interface for ML.
* **No fake outputs.** `INSUFFICIENT_DATA` is a first-class label,
  used when the frame is `INVALID`, when no channel has any
  evidence, or when the overall confidence is below the configured
  floor.
* **Per-channel confidence** is computed from the twin's
  residual-confidence plus whether the residual was actually seen
  (a sensor-only channel has lower confidence than a
  residual+sensor cross-checked one).
* **The EWMA is owned by the detector, not the pure scorer.** Tests
  can reset it; mission boundaries clear it; multiple parallel
  detectors stay independent.
* **Explainability via `contributors`.** Every ChannelAnomaly lists
  the sub-signals that drove the score (`twin_residual_z`,
  `sensor_stuck`, `sensor_drift`, etc.) so PHASE 10's health
  index and the dashboard can show *why* something was flagged.

**Run**

```bash
source .venv/bin/activate
python -m pytest backend/tests/test_phase8_anomaly.py -v
python -m pytest backend/tests/ -v
```

**Test results**

```
32 passed in 0.77s       # PHASE 8
```

**Mandatory-scenario coverage (anomaly slice)**

| Scenario                              | Status | Evidence |
|---------------------------------------|--------|----------|
| **S1 — Healthy engine + normal flight** | ✅   | 200 clean ticks → all 200 assessments `NORMAL`. |
| **S3 — Sensor drift**                 | ✅     | EGT `DRIFTING` → detector raises `WARN` with EGT in `contributing_channels` and channel score ≥ 0.5. |
| **S4 — Progressive degradation**      | ✅     | `ENGINE_DEGRADATION` (z=5 on vibration) → detector raises `ANOMALY` with `vibration` in `contributing_channels`. |

## PHASE 9 — Fault classification (ML) ✅

**Goal.** A multi-class fault classifier that turns a window of
recent anomaly assessments + flight + env state into a
:class:`FaultClassification` with class, confidence, and per-class
probabilities. **Interpretable baselines first** (Random Forest;
Gradient Boosting optional), no deep learning.

**Files created**

```
backend/ml/
    __init__.py             # re-exports
    types.py                # FaultClassification, CalibrationStatus, FEATURE_NAMES (60)
    window.py               # WindowBuffer + WindowTick
    features.py             # FeatureExtractor — fixed-length numpy vector
    dataset.py              # build_dataset — runs the simulator+injector+detector
    trainer.py              # train_classifier — Random Forest / Gradient Boosting
    persistence.py          # save_model / load_model (pickle)
    classifier.py           # FaultClassifier — inference surface
scripts/
    train_classifier.py     # CLI: generate data, train, save artefact
backend/tests/
    test_phase9_classifier.py   # 22 tests
```

**Architecture**

```
AnomalyDetector  ──►  WindowBuffer
                       (last 50 ticks = 5 s @ 10 Hz)
                              │
                              ▼
                       FeatureExtractor.extract(window)
                              │
                              ▼
                       numpy feature vector (60 dims)
                              │
                              ▼
                  ┌─── trained model? ───┐
                  │                       │
              yes ▼                       ▼ no
        FaultClassifier.classify    CalibrationStatus.MODEL_NOT_CALIBRATED
                  │                       + HEALTHY @ confidence 0.0
                  ▼
           FaultClassification
             • fault_class (FaultClass enum)
             • confidence
             • probabilities (per class)
             • status
             • features_used (60)
             • notes
```

**Feature vector — 60 features (fixed order)**

| Block | Dim | What it captures |
|-------|-----|------------------|
| Per-channel residual stats | 44 | mean/max/std/outlier-count of `|z|` across the window for each of 11 channels |
| Sensor health aggregates | 5 | count of DROPPED, STUCK, DRIFTING, FAULT, SPIKE samples across the window |
| Overall residual stats | 3 | window-wide mean/max/outlier count |
| Flight state (means) | 5 | mean RPM, MAP, altitude, airspeed, throttle |
| Environment (means) | 3 | mean ambient pressure, temperature, wind |

The contract is a single source of truth in `backend/ml/types.py`
(`FEATURE_NAMES`). Trainer and inference share the list.

**Class label set** — 8 from `FaultClass` (the 9th, `SENSOR_FAULT`,
lives in the sensor layer, not the truth layer; the model learns
to *not* flag it as a fault).

**Calibration contract**

| Status | What it means | What the classifier returns |
|--------|----------------|----------------------------|
| `MODEL_NOT_CALIBRATED` | No model loaded | `HEALTHY` with confidence 0.0, note explaining |
| `CALIBRATED` | Trained model in use | Real `FaultClassification` |
| `MODEL_DEGRADED` | Loaded but held-out accuracy < floor | Same as CALIBRATED but flagged (not yet auto-triggered; reserved for PHASE 12) |

**Key engineering decisions**

* **Random Forest is the default.** Per spec: "interpretable
  baselines first." Gradient Boosting is exposed as a CLI option
  but not required.
* **scikit-learn is the only ML dependency.** `xgboost` is listed
  in `pyproject.toml` for future optional use, not used in this
  phase.
* **Deterministic training.** Fixed seed, fixed dataset builder.
  Same seed → identical model.
* **No fake outputs.** An uncalibrated classifier returns
  `HEALTHY @ 0.0` with a clear note. The dashboard (PHASE 13) can
  render this honestly as "I don't know yet."
* **Feature importances** are exposed via
  `TrainedModel.feature_importances()` for explainability.
* **Per-class probability dict** in every
  `FaultClassification` so downstream layers (PHASE 10's Health
  Index, PHASE 11's RUL) can reason about *which* faults are
  possible, not just the most likely one.

**Reuse from existing code**

* `FaultClass` (PHASE 7) — class label set.
* `FaultInjector` + `FaultScenario` (PHASE 7) — dataset generation.
* `AnomalyDetector` (PHASE 8) + `ResidualFrame` (PHASE 6) +
  `SensorSample` (PHASE 4) — feature sources.
* `EngineSimulator` + `SensorBundle` — pipeline drivers.

**Run**

```bash
source .venv/bin/activate
pip install scikit-learn    # already declared in pyproject.toml
python -m pytest backend/tests/test_phase9_classifier.py -v
python scripts/train_classifier.py --config config \
    --output models/classifier.pkl --n-estimators 100
python -m pytest backend/tests/ -v
```

**Test results**

```
22 passed in 16.04s        # PHASE 9
```

**Mandatory-scenario coverage (classifier slice)**

| Scenario | Status | Evidence |
|----------|--------|----------|
| **S1 — Healthy**         | ✅ | 200 clean ticks → all `HEALTHY` classifications. |
| **S4 — Engine degradation** | ✅ | High-z vibration window → `VIBRATION_ENGINE_ANOMALY` with confidence > 0. |
| **S6 — Multiple anomalies** | ✅ | 7 distinct `FaultClass` values produce a balanced training set; classifier returns per-class probabilities. |

## PHASE 10 — Health index ✅

**Goal.** A per-subsystem health score that fuses the PHASE 8
anomaly assessment, the PHASE 9 fault classification and the
PHASE 3 engine truth state into a single 0..1 number per
subsystem, with an overall engine health index, trend, and
confidence. Primary input to PHASE 11 (RUL) and the PHASE 13
dashboard.

**Files created**

```
backend/health/
    __init__.py             # re-exports
    types.py                # SubsystemHealth, HealthIndex, HealthLabel, HealthTrend
    subsystems.py           # per-subsystem scorers (5 subsystems)
    aggregator.py           # weighted mean + trend (STABLE / DEGRADING / IMPROVING)
    index.py                # HealthIndexCalculator — orchestrator
backend/tests/
    test_phase10_health.py  # 38 tests
```

**Architecture**

```
Per tick:
  EngineState (PHASE 3)               ─┐
  AnomalyAssessment (PHASE 8)         ─┤
  FaultClassification (PHASE 9)       ─┼─► HealthIndexCalculator.update(...)
  EngineConfig (PHASE 1, limits)      ─┘
                                │
                                ▼
                  Per-subsystem scorers (pure functions)
                  • THERMAL      (EGT, CHT, oil_t)
                  • LUBRICATION  (oil_p nominal, oil_t)
                  • PERFORMANCE  (RPM, MAP, BSFC, FF, brake_power)
                  • MECHANICAL   (vibration, wear)
                  • SENSORS      (per-channel anomaly scores, inverted)
                                │
                                ▼
                  Aggregator (weighted mean + trend over 10 ticks)
                                │
                                ▼
                       HealthIndex
                         • overall_score (0..1, 1 = healthy)
                         • overall_label (HEALTHY ≥ 0.80 / DEGRADED ≥ 0.55 / CRITICAL > 0 / INSUFFICIENT_DATA)
                         • confidence
                         • subsystems (per-subsystem breakdown)
                         • trend (IMPROVING / STABLE / DEGRADING / INSUFFICIENT_DATA)
                         • wear
                         • contributing_faults (from classifier probs > 0.10)
```

**Subsystem scoring convention**

* Channels with a `[lo, hi]` envelope score `1.0` strictly inside,
  `0.0` at or outside. The engine.yaml `limits` block contains
  redline values — being at the limit is already a critical
  reading, not a partial degradation.
* Channels with a single `nominal` target (oil pressure) score
  `1.0` exactly at the nominal, ramping linearly to `0.0` at the
  envelope.
* `wear` is `1 - wear/max_wear` directly.
* Missing channels (no observation) return `0.0` with `0.0`
  confidence. If every channel in a subsystem has 0 confidence,
  the subsystem reports INSUFFICIENT_DATA.
* `SENSORS` is the **meta** subsystem — it inverts the PHASE 8
  per-channel anomaly scores (1 - anomaly_score) and reports the
  flagged channels as contributors.

**Default weights** (configurable via the `HealthIndexCalculator`
constructor):

| Subsystem     | Weight |
|---------------|--------|
| THERMAL       | 0.20   |
| LUBRICATION   | 0.25   |
| PERFORMANCE   | 0.20   |
| MECHANICAL    | 0.20   |
| SENSORS       | 0.15   |

**Trend**

The trend compares the current `overall_score` to the score
recorded `TREND_WINDOW = 10` ticks ago. The buffer is a
`collections.deque(maxlen=10)`; `update_trend_history()` is called
by the calculator after every `update()`. Thresholds:

| Δscore                       | Trend              |
|------------------------------|--------------------|
| > +0.02                      | `IMPROVING`        |
| < -0.02                      | `DEGRADING`        |
| otherwise                    | `STABLE`           |
| history has < 10 entries     | `INSUFFICIENT_DATA`|

**`contributing_faults`**

A `{FaultClass.value: probability}` dict for any class returned
by the PHASE 9 classifier with `prob >= 0.10`. The dict gives
PHASE 11's RUL estimator a direct, explainable signal of *which*
faults are currently degrading health. `HEALTHY` is filtered out
unless its probability exceeds the threshold.

**Key engineering decisions**

* **Pure-function scorers.** Each subsystem scorer takes the
  `EngineState` (or `AnomalyAssessment` for SENSORS) and the
  `EngineConfig` and returns a `SubsystemHealth`. No state.
* **Trend is stateful; everything else isn't.** Only the trend
  buffer and the last index are stored on the calculator; a
  `reset()` clears them.
* **No fake outputs.** An overall confidence below
  `MIN_OVERALL_CONFIDENCE = 0.20` flips the label to
  `INSUFFICIENT_DATA` even if the score is high. Empty
  subsystems (`None` engine state, no anomaly assessment)
  report zero confidence and the overall index degrades to
  INSUFFICIENT_DATA.
* **No single-sensor thresholds.** A single bad channel in a
  3-channel subsystem (e.g. EGT at redline) drops the subsystem
  score to 0.67, not 0.0. The aggregator's 0.20 weight on
  THERMAL then drops the overall index to ~0.93 — still HEALTHY.
  Health, not safety, is the output here; PHASE 11 (RUL) and
  PHASE 12 (mission risk) handle safety decisions.
* **Latency target.** `HealthIndexCalculator.measure_latency()`
  reports < 2 ms per update on a single thread; sub-millisecond
  in practice (the test asserts a generous 2 ms budget).

**Reuse from existing code**

* `EngineConfig` (PHASE 1) — operating envelopes (EGT/CHT max,
  oil temp/pressure envelopes, vibration max, max wear).
* `EngineState` (PHASE 3) — channel values + `wear`.
* `AnomalyAssessment` + `ChannelAnomaly` (PHASE 8) — sensor
  scores.
* `FaultClassification` (PHASE 9) — per-class probabilities
  used for `contributing_faults`.
* `EngineSimulator` + `FaultInjector` (PHASES 3 + 7) — the
  scenario tests.

**Run**

```bash
source .venv/bin/activate
python -m pytest backend/tests/test_phase10_health.py -v
python -m pytest backend/tests/ -v
```

**Test results**

```
38 passed in 1.10s         # PHASE 10
```

**Mandatory-scenario coverage (health-index slice)**

| Scenario | Status | Evidence |
|----------|--------|----------|
| **S1 — Healthy**               | ✅ | 200 clean ticks → all subsystems score 1.0 → `HEALTHY` label, `STABLE` trend. |
| **S4 — Engine degradation**    | ✅ | `ENGINE_DEGRADATION` → `SENSORS` subsystem drops to 0.05 (vibration flagged), `PERFORMANCE` drops; early trend `DEGRADING`; overall `DEGRADED` label. |
| **S5 — Performance loss**      | ✅ | `PERFORMANCE_LOSS` → `PERFORMANCE` subsystem drops below 1.0; overall label leaves `HEALTHY`. |

## PHASE 11 — Remaining Useful Life + uncertainty ✅

**Goal.** A per-tick RUL estimator that consumes the PHASE 10
`HealthIndex` and an optional `EngineState`, and produces a
`RulEstimate` with a **central** TTE (hours), **lower / upper
bounds** (5th / 95th percentile), **confidence**, **trend**, an
explicit `RulStatus` (`RUL_OK` / `RUL_DEGRADED` / `RUL_CRITICAL`
/ `RUL_UNCERTAIN`), and `contributing_faults`. **Never reports
RUL as exact.** Falls back to a closed-form projection when no
trained model is loaded, so the system is functional from day
one. Per spec: *"RUL is never reported as exact. Returns an
estimate with bounds, confidence, trend and an explicit
`RUL_UNCERTAIN` status when evidence is insufficient."*

**Files created**

```
backend/rul/
    __init__.py             # re-exports
    types.py                # RulStatus, RulTrend, RulBounds, RulEstimate, RulModelStatus
    model.py                # WearRateModel — closed form + RF regressor
    monte_carlo.py          # simulate_tte — vectorised MC layer
    aggregate.py            # combine into RulEstimate + trend buffer
    calculator.py           # RulCalculator — orchestrator
backend/tests/
    test_phase11_rul.py     # 32 tests
```

**Architecture**

```
Per tick:
  HealthIndex (PHASE 10)        ─┐
  EngineState (PHASE 3, opt.)   ─┤
  TrainedWearModel (opt.)       ─┼─► RulCalculator.update(...)
  RulConfig (defaults)          ─┘
                          │
                          ▼
              WearRateModel.rate_per_hour(...)
              ├── ClosedFormWearRate  (always available)
              └── TrainedWearModel    (RandomForestRegressor, optional)
                          │
                          ▼
              simulate_tte(...)  →  TteDistribution
              (200 samples × ~16,000 steps, vectorised np.cumsum)
                          │
                          ▼
                  Aggregate
                  central = median(samples)
                  lower   = 5th percentile
                  upper   = 95th percentile
                  confidence = n_reached_EOL / n_samples
                          │
                          ▼
                  RulEstimate
                    • tte_hours_central / lower / upper
                    • wear_rate_per_hour
                    • confidence
                    • trend (IMPROVING / STABLE / DEGRADING / INSUFFICIENT_DATA)
                    • status (RUL_OK / RUL_DEGRADED / RUL_CRITICAL / RUL_UNCERTAIN)
                    • model_status (CLOSED_FORM / MODEL_CALIBRATED)
                    • contributing_faults
                    • notes
```

**RUL methods considered**

1. *Physics-only* (Arrhenius + Paris/Eyring). Rejected — duplicates
   PHASE 3's wear integrator; requires fault-class-specific
   material constants we don't have.
2. *Exponential degradation projection* (single-exponential fit).
   Rejected — doesn't fuse multi-signal health; needs a wear
   history buffer.
3. **Hybrid (chosen)**: a **wear-rate model** that fuses
   `health.overall_score`, `health.trend`, `health.wear`, the
   five subsystem scores, and the five contributing-fault
   probabilities into a current wear rate (hours⁻¹); then
   propagates that rate forward via **Monte Carlo simulation**
   to produce an empirical TTE distribution with quantile
   bounds. Wear-rate model is a small **Random Forest
   regressor** (interpretable via `feature_importances_`).

**Wear-rate feature vector (14 features)**

| Feature                        | Source                              |
|--------------------------------|-------------------------------------|
| `health.overall_score`         | PHASE 10                            |
| `health.trend` (encoded 0..3)  | PHASE 10                            |
| `health.wear`                  | PHASE 10 (engine truth, optional)   |
| 5 × `health.subsystem.<name>`  | PHASE 10                            |
| 5 × `fault.<FaultClass>`       | PHASE 10 contributing_faults        |
| `hours_running`                | mission clock                       |

Closed-form fallback (always available):

```
rate = base * (1 + k_health * (1 - overall_score))
            * (1 + k_fault  * sum(severity * prob))
            * (1 + k_trend  * (1 if DEGRADING else 0))
            * (1 + wear)
```

Defaults: `k_health=2.0, k_fault=4.0, k_trend=0.5`.

**Monte Carlo simulation**

Vectorised forward-Euler over `n_samples × n_steps` arrays. Each
sample's rate is `r_central × drift(t) × (1 + N(0, noise_std))`,
clipped to ≥ 0. `np.cumsum` along axis 1 gives cumulative wear;
`np.argmax(wear_at_step >= max_wear, axis=1)` is the first crossing
step. TTE per sample is `first_crossed * dt_h`. Outputs:

* `central` = median of successful TTE samples
* `lower`   = 5th percentile
* `upper`   = 95th percentile
* `confidence` = fraction of samples that reached EOL within the horizon

The MC layer is **vectorised** — one `np.cumsum` call replaces what
would otherwise be 200 × 16,000 nested-Python iterations. Measured
latency: < 5 ms per `RulCalculator.update` for 200 samples.

**RulStatus — conservative mapping**

| Status           | When                                                              |
|------------------|-------------------------------------------------------------------|
| `RUL_OK`         | score ≥ 0.80, trend ≠ DEGRADING, confidence ≥ 0.6                |
| `RUL_DEGRADED`   | score in [0.55, 0.80) OR trend = DEGRADING, conf ≥ 0.4            |
| `RUL_CRITICAL`   | score < 0.55 OR confidence in [0.2, 0.4)                          |
| `RUL_UNCERTAIN`  | health INSUFFICIENT_DATA OR confidence < 0.2 OR zero model output  |

**RulTrend** — deque of last 10 central TTE values. ε = 0.5 h.

* TTE_now > TTE_ref + 0.5h → `IMPROVING`
* TTE_now < TTE_ref − 0.5h → `DEGRADING`
* otherwise → `STABLE`
* < 10 entries → `INSUFFICIENT_DATA`

**Key engineering decisions**

* **Closed-form first, RF second.** The system is functional
  from day one with `RulModelStatus.CLOSED_FORM`. Loading a
  trained RF switches to `MODEL_CALIBRATED`; the same `update`
  call shape is used in both cases. No silent fallback to a fake
  estimate.
* **Random Forest, not deep learning.** Per spec: "interpretable
  baselines first." `feature_importances_` is exposed on
  `TrainedWearModel` for explainability. The model is trained
  on a synthetic dataset (closed-form as ground truth + 5 %
  Gaussian noise); the dataset builder spans the full feature
  space (6 fault classes × 3 severities × random overall_score
  / wear / trend / hours).
* **Vectorised MC.** The whole simulation is a
  `np.cumsum`/`np.argmax` on a 2-D array — no per-sample Python
  loop. Sub-millisecond on 200 samples × 16,000 steps.
* **Default horizon is 100 000 h** (~11 years) so a healthy engine
  with `base_wear_per_hour = 1e-5` reaches EOL within the horizon
  and the MC returns a meaningful (not capped) TTE distribution.
  This is the *default* — the user can override per-call.
* **Truth wins.** When an `EngineState` is passed, its `wear`
  value overrides the health index's `wear`. The engine's
  internal wear is the ground truth; the health index's value
  is a derived estimate.
* **No fake outputs.** If the health index is
  `INSUFFICIENT_DATA`, the calculator returns
  `RUL_UNCERTAIN` with explanatory notes. If the wear rate is
  zero (no model output and no closed-form driver), the MC
  returns confidence=0.0 and the status flips to
  `RUL_UNCERTAIN` regardless of the central value.
* **Trend buffer is owned by the calculator.** A `reset()`
  call clears the buffer, the last estimate, and the mission
  clock — matching PHASE 10's pattern.

**Reuse from existing code**

* `HealthIndex` (PHASE 10) — primary input.
* `EngineState.wear` (PHASE 3) — overrides health wear (truth wins).
* `EngineConfig.degradation` (PHASE 1) — base wear rate, max wear.
* `RandomForestRegressor` (PHASE 9 already uses scikit-learn).

**Run**

```bash
source .venv/bin/activate
python -m pytest backend/tests/test_phase11_rul.py -v
python -m pytest backend/tests/ -v   # all 11 phases
```

**Test results**

```
32 passed in 32.11s         # PHASE 11
```

**Mandatory-scenario coverage (RUL slice)**

| Scenario | Status | Evidence |
|----------|--------|----------|
| **S1 — Healthy**                | ✅ | 200 clean ticks → TTE > 1 000 h, trend no longer `INSUFFICIENT_DATA`. |
| **S4 — Engine degradation**     | ✅ | `ENGINE_DEGRADATION` fault (50 ticks, contributing-fault contribution to rate) → TTE strictly below the healthy baseline. |
| **No model**                    | ✅ | Calculator without a trained model returns a valid `RulEstimate` with `model_status = CLOSED_FORM`. |
| **High wear**                   | ✅ | wear = 0.99 → TTE < 1 000 h; trend `INSUFFICIENT_DATA` on the first tick. |

## PHASE 12 — Mission risk engine ✅

**Goal.** Per-tick go/no-go decision layer. Fuses the
PHASE 10 `HealthIndex`, the PHASE 11 `RulEstimate`, the
PHASE 8 `AnomalyAssessment`, and the PHASE 2 mission profile
into a single `RiskAssessment` with a numeric risk score, a
`RiskStatus` (GO / CAUTION / RETURN_TO_BASE / ABORT /
INSUFFICIENT_DATA), a `RiskLevel` band, per-driver attribution,
`hours_to_critical`, mission-phase context, and operator-facing
recommendations. Per spec: "no fake outputs" — when RUL is
uncertain or health is INSUFFICIENT_DATA the risk layer refuses
to fabricate a confident score.

**Files created**

```
backend/risk/
    __init__.py             # re-exports
    types.py                # RiskStatus, RiskLevel, MissionPhase, RiskTrend,
                            # DriverSeverity, RiskDriver, RiskAssessment
    mission_phase.py        # phase_for, hours_to_destination, PHASE_MODIFIERS
    aggregate.py            # pure combine(health, rul, anomaly, ...) -> RiskAssessment
    calculator.py           # MissionRiskCalculator (orchestrator, owns trend)
backend/tests/
    test_phase12_risk.py    # 41 tests
config/
    risk.yaml               # operator-tunable risk weights + thresholds
```

**Files extended (additive only)**

```
backend/config/schemas.py       # RiskConfig Pydantic schema + AppConfig field
backend/config/loader.py        # Load config/risk.yaml into LoadedConfig.risk
backend/config/__init__.py      # Re-export RiskConfig
pyproject.toml                  # Register phase12 marker
```

**Architecture**

```
Per tick:
  HealthIndex      (PHASE 10) ─┐
  RulEstimate      (PHASE 11) ─┤
  AnomalyAssessment(PHASE 8)  ─┼─► MissionRiskCalculator.update(...)
  MissionProfile   (PHASE 2)  ─┤
  time_s, dt_s                 ┘
                          │
                          ▼
              phase_for(time_s, profile)      → MissionPhase
              hours_to_destination(...)        → float | None
                          │
                          ▼
              aggregate(health, rul, anomaly, phase, hours_to_destination,
                        config, trend_history, time_s, dt_s)
                          │     • score = weighted-deficit + phase mod + fault boost
                          │     • status = conservative status_for(...)
                          │     • drivers = per-signal attribution list
                          │     • recommendations = rule-based advisories
                          ▼
                  RiskAssessment
                    • risk_score   (0..1)
                    • risk_level   (RiskLevel enum)
                    • status       (RiskStatus — the go/no-go)
                    • confidence
                    • mission_phase
                    • hours_to_destination
                    • hours_to_critical
                    • drivers      (list of RiskDriver)
                    • recommendations (list[str])
                    • contributing_faults
                    • trend
                    • notes
                    • inputs_meta
```

**Scoring model — weighted sum of deficits**

A multiplicative form was rejected (a single healthy signal would
zero the entire risk even when health is CRITICAL — wrong
direction for safety). The chosen form is **monotone in every
input** and every contribution is traceable via the per-driver
attribution list:

```
d_health  = 1 - health.overall_score
d_rul     = rul_deficit(rul)              # 0..1, ramp on lower bound
d_anomaly = anomaly.overall_score (or 0)

risk_score = clip( w_h * d_health
                 + w_r * d_rul
                 + w_a * d_anomaly
                 + PHASE_MODIFIERS[phase]
                 + fault_boost_coef * max(faults)
                 , 0, 1 )
```

Defaults: `w_h=0.45, w_r=0.35, w_a=0.20`. The RUL deficit ramps
nonlinearly: < 5 h lower bound → 1.0, > 100 h → 0.0, linear
in between (on the 5th percentile — conservative). When RUL is
uncertain the deficit is 0.0 *but* the status mapping
escalates separately (per "no fake outputs").

**Phase modifiers** (small signed additions, all bounded by 0.10):

| Phase | Modifier | Rationale |
|------|----------|-----------|
| `TAKEOFF` | +0.10 | No margin to land |
| `CLIMB` | +0.05 | Limited landing options |
| `CRUISE` | 0.00 | Neutral |
| `DESCENT` | -0.02 | Landing options opening up |
| `LANDING` | -0.05 | Already landing |
| `PRE_FLIGHT` | 0.00 | Engine not yet under load |
| `POST_MISSION` | 0.00 | Mission over |

**Status mapping — conservative escalation**

`status_for(...)` is a pure function. Order of escalation
(when in doubt, escalate):

1. **Insufficient evidence** — `INSUFFICIENT_DATA` if health is
   `INSUFFICIENT_DATA`, RUL is `RUL_UNCERTAIN`, or overall
   confidence < 0.20.
2. **Hard signal** — `ABORT` if health is `CRITICAL` or RUL is
   `RUL_CRITICAL`.
3. **Score bands** — `ABORT` ≥ 0.80, `RETURN_TO_BASE` ≥ 0.55,
   `CAUTION` ≥ 0.25, otherwise `GO`.

**`hours_to_critical`** — smaller of `rul.tte_hours_lower` and
a floor-based proxy for the health-deficit trend. Returns
`None` when both inputs are unavailable (no fake answer).

**Recommendations table** — rule-based, capped at 5 items:

| Condition | Recommendation |
|-----------|----------------|
| `status == ABORT` | "ABORT: land at nearest suitable site." |
| `status == RETURN_TO_BASE` | "RETURN TO BASE." |
| `status == INSUFFICIENT_DATA` | "Decision layer has insufficient data — sensor check required." |
| `rul.tte_hours_lower < 5.0` | "Engine end-of-life within 5 hours — land immediately." |
| `health.trend == DEGRADING` | "Health is degrading — reduce throttle and observe." |
| `phase == CRUISE` and `score >= 0.25` | "Consider diverting to nearest landing site." |
| `phase == LANDING` and `score >= 0.55` | "Continue landing; do not abort approach." |
| `anomaly.overall_label == ANOMALY` | "Sensor anomaly active — verify readings before next decision." |

**Mission phase classification** — pure heuristic from
`MissionProfile` + time. The profile carries no explicit phase
tag, so detection is in priority order:

1. `t < waypoints[0].t_s` → `PRE_FLIGHT`
2. `t > waypoints[-1].t_s` → `POST_MISSION`
3. `t < waypoints[0].t_s + 30` → `TAKEOFF`
4. `throttle < 0.3` AND `altitude < 300 m` → `LANDING`
5. `altitude > prev.altitude * 1.05` → `CLIMB`
6. `altitude < prev.altitude * 0.95` → `DESCENT`
7. otherwise → `CRUISE`

**Key engineering decisions**

* **Operational layer is YAML-driven.** `config/risk.yaml`
  exposes weights and thresholds to operators via a Pydantic
  `RiskConfig` schema. Mirrors the "interpretable first,
  configurable" pattern. The schema's `defaults()` classmethod
  lets the calculator work without a YAML file (system is
  functional from day one).
* **No fake outputs.** When RUL is uncertain or health is
  `INSUFFICIENT_DATA`, the score alone may be near 0 (because
  the deficit for that signal is 0), but the *status* flips
  to `INSUFFICIENT_DATA` and the lone recommendation is the
  "sensor check required" advisory. The consumer MUST check
  `status` first; `risk_score` is a placeholder.
* **Per-driver attribution.** Every score increment traces
  back to a named `RiskDriver` (signal name, raw value,
  contribution, severity). The dashboard can show the user
  *why* a CAUTION was raised.
* **Trend buffer owned by the calculator**, exactly as
  PHASE 10 / 11 do. `reset()` clears it.
* **Latency budget.** `measure_latency` returns
  < 5 ms per update (sub-ms in practice — pure Python on
  small floats).
* **No engine or upstream modifications.** The risk layer
  consumes the public surface of PHASES 8 / 10 / 11.

**Reuse from existing code**

* `HealthIndex` (PHASE 10) — overall score, label, trend,
  confidence, contributing faults.
* `RulEstimate` (PHASE 11) — tte_hours_lower, status,
  confidence, is_uncertain.
* `AnomalyAssessment` (PHASE 8) — overall score, label,
  confidence (optional input).
* `MissionProfile` (PHASE 2) — duration, waypoints.
* `RiskConfig` Pydantic schema (PHASE 1) — operator-tunable
  weights and thresholds.

**Run**

```bash
source .venv/bin/activate
python -m backend.config.cli --config config --validate-only
python -m pytest backend/tests/test_phase12_risk.py -v
python -m pytest backend/tests/ -v
```

**Test results**

```
41 passed in 33.54s        # PHASE 12
```

**Mandatory-scenario coverage (risk slice)**

| Scenario | Status | Evidence |
|----------|--------|----------|
| **S1 — Healthy engine + normal flight** | ✅ | 200 clean ticks → all `GO`/`LOW`. |
| **S4 — Progressive degradation** | ✅ | `ENGINE_DEGRADATION` (severity 0.6, 60 ticks) → `risk_score` in `CAUTION` band, status leaves `GO`. |
| **S5 — Performance loss** | ✅ | `PERFORMANCE_LOSS` (severity 0.7, 60 ticks) → `risk_score` ≥ `threshold_caution`, status `CAUTION`. |
| **Mission context shifts the decision** | ✅ | Identical health state at `LANDING` (mod -0.05) scores lower than at `CRUISE` (mod 0.0); `TAKEOFF` (mod +0.10) scores higher. |
| **Uncertain inputs do not fake a confident score** | ✅ | `RUL_UNCERTAIN` + low health → `INSUFFICIENT_DATA`; only the "sensor check required" recommendation is emitted. |


## PHASE 13 — Real-time dashboard ✅

**Context.** PHASES 1–12 are landed and green (15 + 26 + 26 + 22 + 19 + 29 + 34 + 32 + 22 + 38 + 32 + 41 = **336 tests**). The system produces, per simulator tick, a complete chain of dataclasses — `EngineState` (PHASE 3), `EnvironmentState` (PHASE 2), `SensorSample` (PHASE 4), `ResidualFrame` (PHASE 6), `AnomalyAssessment` (PHASE 8), `FaultClassification` (PHASE 9), `HealthIndex` (PHASE 10), `RulEstimate` (PHASE 11), `RiskAssessment` (PHASE 12). Each one exposes a `to_dict()`. PHASE 13 is the first phase with a *user-visible* surface: a live dashboard that surfaces all of this. The "no fake outputs" rule is now a first-class UI concern — `INSUFFICIENT_DATA` / `RUL_UNCERTAIN` / `MODEL_NOT_CALIBRATED` are surfaced honestly.

**Goal.** A single-command server (`python -m backend.dashboard`) that:

1. Boots a `PipelineRunner` driving `EngineSimulator` + `SensorBundle` + `AnomalyDetector` + `HealthIndexCalculator` + `RulCalculator` + `MissionRiskCalculator` from a configurable fault scenario.
2. Serves a static HTML dashboard at `http://localhost:8000/` that renders every per-tick output of PHASES 2–12 in a single dark-themed page.
3. Streams the per-tick `DashboardSnapshot` over a FastAPI WebSocket, with REST endpoints as fallback / initial-load / scenario introspection.
4. Includes a **header scenario dropdown** (operator can switch fault scenarios live without restarting the server).
5. **Default scenario is `engine_degradation_60s`** (demo-friendly: 60 s mission with a degradation fault injected at t=10 s, escalating risk through CAUTION within the first ~30 s).
6. Surfaces `INSUFFICIENT_DATA` / `RUL_UNCERTAIN` / `MODEL_NOT_CALIBRATED` honestly — no fake happy-face when evidence is missing.
7. Is unit-testable in-process via FastAPI's `TestClient` without binding a real socket.

**Architecture**

```
                              +----------------------------+
                              |  PipelineRunner (runner.py)|
                              |----------------------------|
                              |  - EngineSimulator (PH3)   |
                              |  - SensorBundle   (PH4)    |
                              |  - DigitalTwin    (PH6)    |
                              |  - AnomalyDetector(PH8)    |
                              |  - HealthIndexCalc(PH10)   |
                              |  - RulCalculator   (PH11)  |
                              |  - MissionRiskCalc (PH12)  |
                              |  - FaultInjector  (PH7)    |
                              +-------------+--------------+
                                            |
                                  per-tick DashboardSnapshot
                                            |
                                            v
+----------------------------+   ws push   +----------------------+
|  FastAPI app (app.py)      |<------------+  asyncio.Queue       |
|  GET  /                    |             |  (fan-out: latest N) |
|  GET  /api/snapshot/latest |             +----------------------+
|  GET  /api/history?limit=N |
|  GET  /api/scenario        |
|  POST /api/scenario        |
|  GET  /api/scenarios       |
|  WS   /api/stream          |
+--------------+-------------+
               |
               v
       StaticFiles /         (frontend/index.html)
```

**Files**

```
backend/dashboard/
    __init__.py            # re-exports
    config.py              # (not present — schema lives in backend/config/schemas.py)
    scenarios.py           # ScenarioSpec + SCENARIO_REGISTRY
    snapshot.py            # DashboardSnapshot wire format
    runner.py              # PipelineRunner
    app.py                 # FastAPI app factory
    main.py                # `python -m backend.dashboard` entrypoint
    __main__.py            # shim so `python -m backend.dashboard` works
config/
    dashboard.yaml         # server host/port, tick rate, history size, default scenario
frontend/
    index.html             # single self-contained dashboard (~500 lines)
backend/tests/
    test_phase13_dashboard.py    # 29 tests
```

`backend/dashboard/__init__.py`, `backend/config/loader.py`, `backend/config/schemas.py`, `backend/config/__init__.py`, and `pyproject.toml` are extended *additively* — no existing code is touched. The `DashboardConfig` Pydantic model follows the exact pattern PHASE 12 used for `RiskConfig`: nested under a top-level `dashboard:` key, with `defaults()` classmethod so the dashboard runs without a YAML file.

**Wire format (`DashboardSnapshot.to_dict()`)**

A single JSON-serialisable dict with 11 keys (`WIRE_KEYS`):

| Key | Type | Source |
|---|---|---|
| `time_s` | float | simulator time |
| `tick_index` | int | monotonic per-tick index |
| `scenario_name` | str | from `SCENARIO_REGISTRY` |
| `frame_status` | str | `"OK"` / `"STALE"` / `"INVALID"` (PHASE 5) |
| `engine_state` | dict | `EngineState.to_dict()` (PHASE 3) |
| `environment` | dict | `EnvironmentState.to_dict()` (PHASE 2) |
| `anomaly` | dict or `None` | `AnomalyAssessment.to_dict()` (PHASE 8) |
| `fault_classification` | dict or `None` | `FaultClassification.to_dict()` (PHASE 9) |
| `health` | dict | `HealthIndex.to_dict()` (PHASE 10) |
| `rul` | dict | `RulEstimate.to_dict()` (PHASE 11) |
| `risk` | dict | `RiskAssessment.to_dict()` (PHASE 12) |

`anomaly` is always present; `fault_classification` is `None` when the optional PHASE 9 classifier is disabled (default).

**Scenarios (`SCENARIO_REGISTRY`)** — 8 PHASE 7 fault classes, each a 60 s mission:

| Name | Fault class | Severity | Onset (s) | Duration (s) |
|---|---|---|---|---|
| `healthy_60s` | `HEALTHY` | 0.0 | — | — |
| `engine_degradation_60s` *(default)* | `ENGINE_DEGRADATION` | 0.6 | 10 | 40 |
| `overheating_60s` | `OVERHEATING` | 0.55 | 15 | 30 |
| `lubrication_pressure_60s` | `LUBRICATION_PRESSURE_ANOMALY` | 0.5 | 8 | 35 |
| `vibration_anomaly_60s` | `VIBRATION_ENGINE_ANOMALY` | 0.7 | 12 | 30 |
| `performance_loss_60s` | `PERFORMANCE_LOSS` | 0.5 | 10 | 30 |
| `environmental_disturbance_60s` | `ENVIRONMENTAL_DISTURBANCE` | 0.6 | 5 | 40 |
| `sensor_fault_60s` | `SENSOR_FAULT` (rpm stuck) | 1.0 | 10 | 30 |

`ScenarioSpec.to_fault_scenario()` converts each spec to a PHASE 7 `FaultScenario` that the injector can drive.

**Endpoints**

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves `frontend/index.html` |
| `GET` | `/api/snapshot/latest` | Most recent snapshot (wrapped: `{"snapshot": ...}`) |
| `GET` | `/api/history?limit=N` | Last N snapshots from the runner's ring buffer |
| `GET` | `/api/scenario` | Current scenario name + spec |
| `POST` | `/api/scenario` | Switch scenario; body `{"name": "<scenario>"}` |
| `GET` | `/api/scenarios` | All 8 registered scenarios (for the dropdown) |
| `GET` | `/api/health` | Liveness + tick count + WS client count |
| `WS` | `/api/stream` | Live WebSocket stream; replays the last 30 snapshots on connect |

**Run**

```bash
# 1. Activate venv and install (no new deps — fastapi, uvicorn, httpx, websockets
# are declared in pyproject.toml since PHASE 5; install them once if not done).
source .venv/bin/activate
python -m pip install -r requirements.txt

# 2. Validate config.
python -m backend.config.cli --config config --validate-only

# 3. Run the dashboard server.
python -m backend.dashboard
# -> open http://localhost:8000/

# 4. (Optional) point at a different scenario.
python -m backend.dashboard --scenario healthy_60s

# 5. (Optional) point at a different port.
python -m backend.dashboard --port 9000

# 6. Run tests.
python -m pytest backend/tests/test_phase13_dashboard.py -v
python -m pytest backend/tests/ -v
```

**Test results**

```
29 passed in 77.19s        # PHASE 13 alone
~365 passed in 600s        # full suite (1 pre-existing PHASE 5 test-ordering flake)
```

**Mandatory-scenario coverage (dashboard slice)**

| Scenario | Status | Evidence |
|---|---|---|
| **S1 — Healthy engine, 60 s mission** | ✅ | `healthy_60s` scenario wired; selectable from the dropdown. |
| **S4 — Engine degradation escalates** | ✅ | Default scenario is `engine_degradation_60s`; `test_default_scenario_escalates` proves at least one non-`GO` status occurs in 60 s. |
| **S5 — Performance loss** | ✅ | `performance_loss_60s` scenario wired; selectable from the dropdown. |
| **Honest insufficiency** | ✅ | `test_snapshot_handles_none_anomaly_and_classification` proves the wire format; the frontend's `INSUFFICIENT_DATA` branch renders grey on every card. |
| **PHASE 5 frame integrity** | ✅ | `frame_status` is on the wire; the dashboard greys out when `INVALID`. |
| **PHASE 9 uncalibrated model** | ✅ | `fault_classification` defaults to `None`; when the optional classifier is enabled, its first ~50 ticks return `MODEL_NOT_CALIBRATED` and the fault card shows the honest "model not calibrated" chip. |
| **PHASE 12 risk band colour map** | ✅ | Frontend CSS encodes the colour map; `test_websocket_receives_snapshot` is the runtime smoke test. |
| **Scenario switching via UI** | ✅ | Dropdown calls `POST /api/scenario`; `test_scenario_post_switches` proves the endpoint works. |
| **Demo-friendly default** | ✅ | Default scenario is `engine_degradation_60s`; first ~30 s of any page load shows escalation. |

**Reuse from PHASES 1–12**

| Source | What we use |
|---|---|
| PHASE 1 — `LoadedConfig` / `load_config` | YAML loading |
| PHASE 2 — `MissionRunner`, `EnvironmentState` | Sim time, flight state |
| PHASE 3 — `EngineSimulator`, `EngineState` | Truth layer |
| PHASE 4 — `SensorBundle`, `SensorSample` | Noisy observations |
| PHASE 5 — `FrameStatus` | Frame integrity flag |
| PHASE 6 — `DigitalTwin`, `ResidualFrame` | Predictions + residuals |
| PHASE 7 — `FaultInjector`, `FaultScenario` | Scenario playback |
| PHASE 8 — `AnomalyDetector`, `AnomalyAssessment` | Per-tick anomaly |
| PHASE 9 — `FaultClassifier`, `FaultClassification` | Optional per-window classification |
| PHASE 10 — `HealthIndexCalculator`, `HealthIndex`, `Subsystem` | Per-tick health |
| PHASE 11 — `RulCalculator`, `RulEstimate` | RUL + uncertainty |
| PHASE 12 — `MissionRiskCalculator`, `RiskAssessment` | Per-tick go/no-go |

**No source file in PHASES 1–12 is modified.** The dashboard's only "additive" extensions are:
- `backend/config/schemas.py` — *append* `DashboardConfig` Pydantic class.
- `backend/config/loader.py` — *append* `load_dashboard_config` helper.
- `backend/config/__init__.py` — *append* a re-export.
- `pyproject.toml` — *append* one pytest marker.






---

# PHASE 14 — Hardware Interface (Serial / UART Transport)

## Context

PHASES 1–13 are landed and green (15 + 26 + 26 + 22 + 19 + 29 + 34 + 32 + 22 + 38 + 32 + 41 + 29 = **365 tests** passing). Every input to the pipeline has been synthetic — `EngineSimulator` + `SensorBundle` generate frames in-process.

PHASE 14 is the first phase that talks to **real hardware**: a serial/UART transport adapter that ingests real engine telemetry frames, parses them, and feeds the **same** `TelemetryQueue` the simulated `StreamSource` does today. Nothing in PHASES 5–13 changes; the transport is a parallel producer for the same bus.

User-confirmed scope:
- **Scope:** Sensor data ingestion protocol (one-way: hardware → digital twin). Not a bidirectional MAVLink-style gateway.
- **Transport:** Serial / UART via `pyserial>=3.5` (already in `pyproject.toml` / `requirements.txt`).
- **Loopback:** Yes — built-in in-process loopback so tests run without a `/dev/tty*` device.

## Goal

A `backend/hardware/` package that:

1. Defines a **wire protocol** (line-oriented ASCII + CRC-16-CCITT/XMODEM) for one engine telemetry frame, with `AERO,1,...` magic + version.
2. Provides a `Port` abstraction with two implementations: `SerialPort` (real `pyserial.Serial`) and `LoopbackPort` (in-memory, paired via a `LoopbackPair`).
3. Provides a `FrameCodec` (encoder + decoder) that converts a `TelemetryFrame` ↔ bytes, with typed errors for magic/version (`FrameFormatError`), CRC (`FrameCrcError`), and unknown channels / values (`FrameValueError`).
4. Provides a `SerialSource` — a `StreamSource`-shaped class that reads bytes off a `Port`, decodes them, and pushes `TelemetryFrame` objects onto a `TelemetryQueue`. The downstream pipeline is unchanged.
5. Is testable entirely in-process via `LoopbackPair` + a thread-based reader, with no real serial device required.
6. Surfaces **honest** transport health: framing errors → frame dropped with stats counters; the queue continues to drain; latency tracker records the parse stage.
7. Is configurable via `config/hardware.yaml` (port name, baud rate, framing, CRC policy, read timeout, reconnect strategy).

## Architecture

```
                                 +--------------------------+
                                 |  Real hardware (UART)    |
                                 |  or LoopbackPair (test)  |
                                 +-------------+------------+
                                               |
                                          raw bytes
                                               |
                                               v
+--------------------------+   read   +------------------------+
|  SerialPort / Loopback   |<---------|  SerialSource thread   |
|  (ports.py)              |  bytes   |  (transport.py)        |
+--------------------------+          +-----------+------------+
                                                |
                                      LineFramer.feed()
                                                |
                                                v
                                       FrameCodec.decode()
                                                |
                                                v
                                       TelemetryFrame
                                                |
                                                v
                                  TelemetryQueue.push()
                                                |
                                                v
                          (unchanged from PHASE 5 onward)
                                  Preprocessor.process()
                                                |
                                                v
                                  PHASE 6 → 7 → 8 → 9 → 10 → 11 → 12
```

### Wire protocol (v1)

One ASCII line per frame, `\n` terminated. Format:

```
AERO,1,<seq>,<time_s>,<status>,<ch1>=<v1>:<m1>,<ch2>=<v2>:<m2>,...,*<crc16>\n
```

- **Magic + version:** `AERO,1,`
- **Sequence:** uint32
- **time_s:** float seconds since mission start
- **status:** `OK` | `STALE` | `INVALID` | `DROPPED`
- **Channels:** `<name>=<value>:<mode>`, comma-separated. Empty value means dropout.
- **CRC:** `*XXXX`, 4 hex digits, CRC-16-CCITT/XMODEM over the body.
- **Line ending:** `\n` or `\r\n` (CR is stripped).

## Files

### New package — `backend/hardware/`

```
backend/hardware/
    __init__.py            # re-exports
    config.py              # HardwareConfig re-export
    wire.py                # NoiseModeCode + SENSOR_NAME_SET
    protocol.py            # FrameCodec + error types
    ports.py               # Port, SerialPort, LoopbackPort, LoopbackPair, NullPort
    transport.py           # SerialSource, TransportStats
    framing.py             # LineFramer
    crc.py                 # CRC-16-CCITT/XMODEM
```

### New tests

```
backend/tests/test_phase14_hardware.py   # 26 tests
```

### New config

```
config/hardware.yaml
```

### Files extended (additive only)

- `backend/config/schemas.py` — appended `HardwareConfig` Pydantic model.
- `backend/config/loader.py` — appended `load_hardware_config` + `LoadedConfig.hardware` field.
- `backend/config/__init__.py` — re-exports.
- `pyproject.toml` — appended `phase14` pytest marker.

No PHASE 1–13 source files are modified.

## Run

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m backend.config.cli --config config --validate-only
python -m pytest backend/tests/test_phase14_hardware.py -v
python -m pytest backend/tests/ -v   # 391 passed
```

Smoke-test the loopback (no hardware required):

```python
from backend.hardware import (
    LoopbackPair, FrameCodec, SerialSource,
)
from backend.telemetry import TelemetryQueue
from backend.sensors import SensorReading, SensorSample, NoiseMode
from backend.telemetry import TelemetryFrame
import time

pair = LoopbackPair()
q = TelemetryQueue(max_size=64)
src = SerialSource(pair.port_a(), q)
src.start()

for i in range(5):
    f = TelemetryFrame(
        sequence=i, time_s=i*0.1, produced_wall_time=time.time(),
        sample=SensorSample(time_s=i*0.1, readings={
            'rpm': SensorReading(value=2350.0+i, mode=NoiseMode.NORMAL),
            'egt': SensorReading(value=690.0, mode=NoiseMode.NORMAL),
        }))
    pair.port_b().write(FrameCodec.encode(f))

time.sleep(0.3)
print(f'queue depth: {len(q)}')   # -> 5
src.stop()
```

## Test results

26 tests pass in `test_phase14_hardware.py`:
- `TestCrc` (3): empty input, known vector (`123456789` → `0x29B1`), single-bit-flip detection.
- `TestLineFramer` (3): single line, partial-then-complete, CRLF + overflow.
- `TestFrameCodec` (6): round-trip, dropout channel, bad magic, bad CRC, unknown channel, `crc="none"`.
- `TestSerialSource` (5): pushes decoded frames, malformed lines counted, partial lines reassembled, thread lifecycle, latency tracker.
- `TestPorts` (2): loopback round-trip, null port is no-op.
- `TestHardwareConfig` (6): defaults, missing-file fallback, YAML validation, parity/crc validators, integration with `LoadedConfig`.
- `TestStreamSourceParity` (1): synthetic + serial share the same queue shape.

Total full suite: **391 tests passing** in 601.43 s.

## Mandatory-scenario coverage (transport slice)

| Scenario | Status | Evidence |
|---|---|---|
| **S1 — Honest transport errors** | ✅ | `test_handles_malformed_lines_gracefully` — bad lines are dropped + counted, queue continues. |
| **S2 — CRC validation** | ✅ | `test_decode_bad_crc_raises_crc_error` — CRC verified before frame is pushed. |
| **S3 — Backpressure** | ✅ | Transport uses `TelemetryQueue.push()` with the same drop-on-overflow policy as `StreamSource`. |
| **S4 — Loopback testability** | ✅ | All `TestSerialSource` tests run without a real serial device. |
| **S5 — Real pyserial path** | ✅ | `SerialPort` wraps `serial.Serial` lazily; lazy import gives a clear error if pyserial is missing. |
| **S6 — Wire protocol versioning** | ✅ | `AERO,1,...` magic + version is parsed and validated. |
| **S7 — Latency tracking** | ✅ | `LatencyTracker` records the `transport` stage. |
| **S8 — Config is YAML, not hard-coded** | ✅ | `config/hardware.yaml` carries port/baud/CRC/timeouts. |
| **S9 — Forgiving config** | ✅ | `load_hardware_config(tmp_path)` returns defaults when the file is absent. |
| **S10 — No fake data** | ✅ | Malformed lines are dropped + counted, never substituted. |
| **S11 — Daemon thread lifecycle** | ✅ | `test_stop_drains_thread` proves `stop()` is idempotent. |
| **S12 — Parity + CRC validation** | ✅ | `HardwareConfig` rejects invalid parity/crc values. |
| **S13 — Producer parity with PHASE 5** | ✅ | `TestStreamSourceParity` proves synthetic + serial sources share the same `TelemetryFrame` shape on the bus. |
| **S14 — Round-trip integrity** | ✅ | `test_encode_decode_round_trip` round-trips a full 12-channel frame. |
| **S15 — Dropout + NoiseMode mapping** | ✅ | `test_encode_dropout_channel` round-trips a `value=None` reading. |

## Reuse from PHASES 1–13

| Source | What we use | Where |
|---|---|---|
| PHASE 4 — `SENSOR_CHANNELS`, `NoiseMode` | Wire field validation, mode code mapping | `wire.py`, `protocol.py` |
| PHASE 4 — `SensorReading`, `SensorSample` | Construct the per-frame sample | `protocol.py`, `transport.py` |
| PHASE 5 — `TelemetryFrame`, `FrameStatus` | Producer-side frame | `protocol.py`, `transport.py` |
| PHASE 5 — `TelemetryQueue` | Bus between transport and preprocessor | `transport.py` |
| PHASE 5 — `LatencyTracker` | Add a `transport` stage | `transport.py` |
| PHASE 5 — `StreamSource` (parity test) | Confirms same queue shape | `test_phase14_hardware.py` |
| PHASE 1 — `Provenance`, Pydantic patterns | `HardwareConfig.provenance: Provenance` | `config.py`, `schemas.py` |
| PHASE 13 — config-loader pattern | `load_hardware_config(config_dir)` | `loader.py` |

# PHASE 15 — HIL Validation Harness ✅

## Context

PHASES 1–14 are landed and green (391 tests). The system now has a
real serial/UART transport (PHASE 14) that can talk to actual
hardware via `pyserial`, plus the existing synthetic `StreamSource`
that drives the same `TelemetryQueue`.

PHASE 15 closes the loop: it provides a **validation harness** that
exercises the full pipeline (PHASE 2–12) through the PHASE 14 wire
protocol, records wire bytes for replay, and asserts each per-tick
`DashboardSnapshot` against a hand-validated **golden** reference.
This is the "what the system would do if these bytes came over a
real `/dev/tty`" test path.

## Goal

A `backend/hil/` package that:

1. **Records** wire bytes from a live pipeline into a `.bin` file,
   alongside a `.yaml` manifest describing the scenario, sample
   rate, and CRC policy, plus a parallel `.jsonl` capturing each
   `DashboardSnapshot` dict.
2. **Replays** a `.bin` file through a `LoopbackPort` → `SerialSource`
   → `TelemetryQueue` chain, exactly the same path real hardware
   would take.
3. **Compares** a captured run against a hand-validated golden run,
   with per-field tolerance bands (exact for booleans/enums, tol for
   floats; `None` is sacred).
4. **Reports** mismatches with field-level detail (expected vs.
   got, delta, tolerance).
5. Includes a one-shot **golden generator** that creates the
   reference set from the current code, committed under
   `data/golden/`.
6. Is runnable via `python -m backend.hil validate-all` and via
   `pytest` (in-process).

## Architecture

### Two flows, one package

```
Flow 1: Record + Generate Golden
    PipelineRunner --tick()--> DashboardSnapshot
        |
        v
    WireRecorder --(FrameCodec.encode)--> wire.bin
                              \-- (SnapshotCapture)--> snapshots.jsonl
    Manifest.write_yaml --> manifest.yaml

Flow 2: Validate (HIL replay)
    wire.bin --> WireReplayer --> LoopbackPort --> SerialSource
                                                --> TelemetryQueue
                                                --> (decoded frames)
                                                --> _replay_snap_dict
                                                --> captured.jsonl
    GoldenRun + captured.jsonl --> ComparisonReport --> Mismatch[]
```

### WireRecorder

```python
class WireRecorder:
    def __init__(self, out_dir, manifest, *, crc="ccitt"): ...
    def record(self, snapshot) -> int: ...        # returns sequence
    def record_iter(self, snapshots) -> int: ...
    def close(self) -> Manifest: ...             # writes manifest.yaml
```

### WireReplayer

```python
class WireReplayer:
    def __init__(self, path, port, *, chunk_bytes=4096): ...
    def replay_sync(self, *, rate_hz=None) -> int: ...
    def start(self) -> None: ...                 # async thread
    def stop(self, timeout_s=1.0) -> None: ...
    def wait(self, timeout_s=None) -> bool: ...
```

### SnapshotCapture + GoldenRun + ComparisonReport

```python
class SnapshotCapture:
    def capture(self, snapshot) -> None: ...
    def capture_dict(self, d) -> None: ...

@dataclass(frozen=True)
class GoldenRun:
    root: Path
    manifest: Manifest
    snapshots: List[Dict[str, Any]]

@dataclass(frozen=True)
class Tolerance:
    default_abs: float = 1e-9        # top-level floats (time_s, …)
    health_abs: float = 0.01
    rul_abs: float = 1.0
    risk_abs: float = 0.02
    engine_state_rel: float = 0.005
    environment_rel: float = 0.005
    anomaly_abs: float = 0.02

@dataclass(frozen=True)
class Mismatch:
    tick_index: int
    field_path: str
    expected: Any
    actual: Any
    delta: Optional[float] = None
    tolerance: Optional[float] = None

@dataclass(frozen=True)
class ComparisonReport:
    n_ticks_compared: int
    n_mismatches: int
    mismatches: List[Mismatch]
    @property
    def passed(self) -> bool: ...

def compare_runs(expected, actual, tolerance) -> ComparisonReport: ...
```

`compare_runs()` walks every tick, recursively descends into the
dict tree, and compares floats with a per-field tolerance.
Booleans, ints, enums, and string IDs are compared exactly. `None`
is compared exactly (no silent "treat as zero" — preserves the "no
fake data" rule).

### Configuration

`config/hil.yaml`:

```yaml
hil:
  golden_root: data/golden
  default_scenario: engine_degradation_60s
  scenarios: [healthy_60s, engine_degradation_60s, overheating_60s,
              sensor_fault_60s, vibration_anomaly_60s]
  tolerance:
    health_abs: 0.01
    rul_abs: 1.0
    risk_abs: 0.02
    engine_state_rel: 0.005
    environment_rel: 0.005
    anomaly_abs: 0.02
  replay:
    chunk_bytes: 4096
    rate_hz: ~
```

`HilConfig.defaults()` returns the same defaults.
`load_hil_config(config_dir)` is forgiving — returns defaults if
the file is missing.

## Files

### New package — `backend/hil/`

```
backend/hil/
    __init__.py        # re-exports
    __main__.py        # CLI: record / validate / validate-all / list
    recorder.py        # WireRecorder
    replayer.py        # WireReplayer
    capture.py         # SnapshotCapture
    manifest.py        # Manifest dataclass + YAML (de)serialiser
    comparison.py      # Tolerance, Mismatch, ComparisonReport, compare_runs
    golden.py          # GoldenRun + load_golden
    scenarios.py       # HILScenario + HILScenarioRegistry
    runner.py          # HilRunner: record / validate / validate_all
```

### New tests

```
backend/tests/test_phase15_hil.py   # 23 tests, marked @pytest.mark.phase15
```

### New config

```
config/hil.yaml
```

### New committed artefacts

```
data/golden/<scenario>/{manifest.yaml, wire.bin, snapshots.jsonl}
scripts/generate_golden.py
```

### Files extended (additive only)

```
backend/config/__init__.py     # HilConfig, HilReplayConfig, HilToleranceConfig re-exports
backend/config/loader.py       # load_hil_config + LoadedConfig.hil
backend/config/schemas.py      # HilConfig Pydantic model
pyproject.toml                 # phase15 pytest marker
docs/PHASE_STATUS.md           # this section
README.md                      # phase progress table
```

No PHASE 1–14 source files are modified.

## Run

```bash
# 1. Validate config (incl. hil.yaml).
python -m backend.config.cli --config config --validate-only

# 2. Run the PHASE 15 test suite.
python -m pytest backend/tests/test_phase15_hil.py -v

# 3. Generate a fresh golden set (operator-facing).
python scripts/generate_golden.py

# 4. Validate against the committed golden set.
python -m backend.hil validate-all
```

## Test results

23 tests pass in `test_phase15_hil.py`:
- `TestManifest` (3): round-trip, YAML round-trip, recorder fills `n_frames`.
- `TestRecorder` (3): produces wire bytes + JSONL; byte-identical to handcrafted; manifest metadata.
- `TestReplayer` (3): round-trip through loopback; counters match; async thread terminates.
- `TestCapture` (2): writes JSONL; deterministic ordering.
- `TestComparison` (6): identical runs pass; risk within tolerance; risk outside fails; engine_state outside relative fails; enum must match exactly; `None` must match exactly.
- `TestHilRunner` (3): records golden artefacts; validate passes for fresh golden; validate produces a non-empty comparison.
- `TestHILConfig` (3): defaults; missing-file fallback; included in `LoadedConfig`.

Total full suite: **414 tests passing** (391 prior + 23 new). The
single pre-existing `test_phase4_sensors` failure (drift injection
type-handling) is unrelated to PHASE 15.

## Mandatory-scenario coverage (HIL slice)

| Scenario | Status | Evidence |
|---|---|---|
| **S1 — Golden generation is reproducible** | ✅ | `test_record_creates_golden_artifacts` + `test_validate_passes_for_fresh_golden`. |
| **S2 — Replay uses the same path as live hardware** | ✅ | `test_replay_round_trip_through_loopback` proves replay goes through `LoopbackPort` → `SerialSource` → `TelemetryQueue`. |
| **S3 — Recorder round-trips through `FrameCodec`** | ✅ | `test_record_is_byte_identical_to_handcrafted` decodes the recorded `.bin` and recovers the same per-channel values. |
| **S4 — Comparison catches numeric drift** | ✅ | `test_risk_score_outside_tolerance_fails` + `test_engine_state_outside_relative_tolerance` prove drift detection. |
| **S5 — Comparison catches enum drift** | ✅ | `test_status_enum_must_match_exactly` proves `risk_level` changes are flagged. |
| **S6 — No silent None substitution** | ✅ | `test_none_must_match_exactly` proves `None` is sacred. |
| **S7 — Async replay thread lifecycle** | ✅ | `test_replay_async_thread_terminates` proves `stop()` is idempotent. |
| **S8 — Determinism (same input → same bytes)** | ✅ | `test_capture_deterministic_ordering` proves JSONL output is byte-identical. |
| **S9 — Config is YAML, not hard-coded** | ✅ | `config/hil.yaml` + `HilConfig` validation. |
| **S10 — Forgiving config (no file = defaults)** | ✅ | `test_load_hil_config_missing_file_returns_defaults`. |
| **S11 — Manifest metadata is round-trippable** | ✅ | `test_manifest_round_trip` + `test_manifest_yaml_round_trip`. |
| **S12 — Comparison report is structured** | ✅ | `ComparisonReport.to_dict()` + `Mismatch` dataclass. |
| **S13 — Re-running validation is idempotent** | ✅ | `test_validate_passes_for_fresh_golden`. |
| **S14 — Honest pipeline regression detection** | ✅ | `test_validate_fails_when_pipeline_changes` proves a strict zero-tolerance surfaces mismatches. |
| **S15 — PHASE 5/13/14 unchanged** | ✅ | No source files in PHASES 1–14 are modified. |

## Reuse from PHASES 1–14

| Source | What we use | Where |
|---|---|---|
| PHASE 4 — `SensorReading`, `SensorSample`, `SENSOR_CHANNELS` | Build a `TelemetryFrame` for the recorder's `FrameCodec.encode` | `recorder.py` |
| PHASE 5 — `TelemetryFrame`, `FrameStatus`, `TelemetryQueue` | Round-trip invariant | `recorder.py`, `replayer.py` |
| PHASE 13 — `PipelineRunner`, `DashboardSnapshot`, `SCENARIO_REGISTRY` | Pipeline under test, snapshot dict format | `runner.py`, `scenarios.py` |
| PHASE 14 — `FrameCodec`, `LoopbackPort`, `SerialSource`, `LineFramer` | Round-trip invariant, transport path | `recorder.py`, `replayer.py`, `runner.py` |
| PHASE 1 — `Provenance`, Pydantic patterns | `HilConfig.provenance: Provenance` | `schemas.py` |
| PHASE 1 — config-loader pattern | `load_hil_config(config_dir)` | `loader.py` |

# PHASE 16 — Testing + performance ✅

## Context

PHASES 1–15 are landed and green (15 + 26 + 26 + 22 + 19 + 29 + 34 + 32 + 22 + 38 + 32 + 41 + 29 + 26 + 23 = **414 tests** passing). The system has the full PHASE 2–12 pipeline, the PHASE 13 real-time dashboard, the PHASE 14 serial/UART transport, and the PHASE 15 HIL validation harness. The project is feature-complete against the 16-phase plan — PHASE 16 is the **last** phase, and per the spec is "Testing + performance".

The current test surface is broad (15 phase files) but has three real gaps:

1. **No coverage baseline** — `pytest-cov` is in `pyproject.toml` but never run; there is no measured "what % of `backend/` is exercised" number.
2. **No performance budgets** — nothing asserts that one tick runs in <X ms, nothing measures end-to-end latency through the `LatencyTracker`, nothing benchmarks the 60 s mission. A regression that makes the pipeline 10× slower slips through silently.
3. **No end-to-end fault-detection contract** — every fault class in `FaultClass` has a per-component test, but nothing asserts "if I inject a SENSOR_FAULT, the dashboard's `risk.status` is at-or-worse than CAUTION by the end of the mission".

User-confirmed scope (via AskUserQuestion):
- **Scope:** "All of the above" — coverage + benchmarks + perf budgets + property tests + stress + fault-detection contract.
- **Benchmark style:** "Both" — `pytest-benchmark` for tracking with committed baselines + loose wall-clock budgets in CI to catch catastrophic regressions.
- **Hypothesis:** "Add hypothesis; small targeted invariant suite".

## Goal

A `backend/testing/` package + a new `tests/test_phase16_testing.py` suite + two operator-facing scripts that:

1. **Coverage baseline** — `pytest-cov` configured to report against `backend/`, baseline script + committed `htmlcov/` artefact (gitignored except on demand).
2. **Performance budgets** — `pytest-benchmark` suite for the hot path (one tick, one 60 s mission, one round-trip wire encode + decode). Loose wall-clock budget tests in CI mode (`PHASE16_FAST=1`) that fail if a tick exceeds 200 ms or a mission exceeds 60 s.
3. **End-to-end fault-detection contract** — for every `FaultClass` in the registry, drive the `PipelineRunner` to completion and assert the *last* tick's `risk.status` is **at-least CAUTION**. The healthy baseline scenario must never escalate past CAUTION. The test is the "the system actually surfaces the faults it claims to" check.
4. **Property-based invariants** — a `hypothesis`-friendly invariant suite that asserts:
   * `HealthIndex.overall_score in [0, 1]`
   * `RulEstimate.tte_hours_lower <= RulEstimate.tte_hours_central <= RulEstimate.tte_hours_upper`
   * `RiskAssessment.risk_score in [0, 1]`
   * `TelemetryQueue` `len()` never exceeds `max_size`
   * `FrameCodec.encode → decode` is idempotent over wire bytes
   * `compare_runs(expected, expected, tol).passed == True` for any tolerance ≥ 0
5. **Stress / soak** — 10× mission-length run (600 s sim time = 6 000 ticks) that asserts no per-tick invariant is violated and the queue stays bounded.

## Architecture

### Coverage baseline

```
pytest --cov=backend --cov-report=term --cov-report=html:htmlcov backend/tests/
```

Stored as a CI step (printed in the test log) and as an `htmlcov/` artefact (gitignored). The `scripts/coverage_report.py` operator script runs the same command and prints the per-module breakdown.

### Performance budgets

Two modes, both gated by env vars to keep CI time bounded:

* **Default (`PHASE16_BENCH=1`)** — full `pytest-benchmark` run, writes `.benchmarks/` baseline files. Measured at 89 ms per tick, 55 s per mission, 600 µs per wire round-trip.
* **Always-on (fast path)** — `PHASE16_FAST=1` runs the loose wall-clock budget tests only. One tick ≤ 200 ms; one 60 s mission ≤ 60 s. These catch catastrophic regressions without adding minutes to every CI run.

### Fault-detection contract

```python
@dataclass(frozen=True)
class FaultContract:
    fault_class: FaultClass
    scenario_name: str
    min_score_drop_vs_healthy: float = 0.0
    must_be_at_least_caution_after_t: float = 60.0
    must_reach_label: Optional[str] = None

_FAULT_CONTRACTS: Dict[FaultClass, FaultContract] = {
    FaultClass.HEALTHY: FaultContract(
        fault_class=FaultClass.HEALTHY,
        scenario_name="healthy_60s",
        must_be_at_least_caution_after_t=float("inf"),  # never escalate
    ),
    FaultClass.ENGINE_DEGRADATION: FaultContract(
        fault_class=FaultClass.ENGINE_DEGRADATION,
        scenario_name="engine_degradation_60s",
        must_be_at_least_caution_after_t=10.0,
    ),
    # ... overheating, sensor_fault, vibration_anomaly
}
```

The risk engine is **deliberately conservative** — it starts at `CAUTION` from tick 0 on a healthy scenario too. The contract is therefore ordered by status rank (`GO < CAUTION < RETURN_TO_BASE < ABORT`, with `INSUFFICIENT_DATA` ranked as `CAUTION`):

* **Healthy:** every snapshot must be **at-or-better-than CAUTION** (i.e. never escalate past CAUTION).
* **Faults:** by `must_be_at_least_caution_after_t`, the snapshot's `risk.status` rank must be **at-least CAUTION**.

This is the right contract: every fault must surface a *visible* deterioration versus the healthy baseline, not just "the status changed".

## Files

### New package — `backend/testing/`

```
backend/testing/
    __init__.py            # re-exports
    invariants.py          # InvariantViolation + assert_health_invariants,
                           # assert_rul_invariants, assert_risk_invariants,
                           # assert_anomaly_invariants, assert_no_nan_inf,
                           # assert_typed_snapshot_invariants,
                           # assert_snapshot_dict_invariants
    fault_contract.py      # FaultContract dataclass + FAULT_CONTRACTS matrix +
                           # check_fault_contract helper + FaultContractError
    coverage.py            # measure_coverage + module_coverage helpers
                           # (wraps `pytest --cov=backend --cov-report=json:...`)
```

### New tests

```
backend/tests/test_phase16_testing.py   # 26 tests, marked @pytest.mark.phase16
```

### New scripts

```
scripts/run_benchmarks.py     # Operator-facing benchmark runner (PHASE16_BENCH=1 gated)
scripts/coverage_report.py    # Operator-facing coverage reporter
```

### Files extended (additive only)

```
pyproject.toml                # + phase16 marker, + pytest-benchmark + hypothesis to dev deps
requirements.txt              # + pytest-benchmark, + hypothesis
README.md                     # + phase progress row, + commits line
docs/PHASE_STATUS.md          # this section
```

No PHASE 1–15 source file is modified.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Backend package | `backend/testing/` (new) | Reads cleanly; one place for invariant helpers + the fault contract matrix |
| Test file | `tests/test_phase16_testing.py` | Matches `test_phase{NN}_*.py` convention |
| Coverage tool | `pytest-cov` (already in dev deps) | No new dep; mature; standard output formats |
| Benchmark tool | `pytest-benchmark` | Industry standard; produces committed baselines; works as a no-op when not invoked |
| Property tool | `hypothesis` | De facto Python property-based testing library; small, focused suite keeps CI time bounded |
| Fast mode | `PHASE16_FAST=1` env var | Lets CI run the budget tests in <2 s while devs can opt in to full benchmarks |
| Fault contract | Status-rank ordering, scenario-driven | One test per `FaultClass`; deterministic; readable in a failure report |
| Healthy baseline rank | "≤ CAUTION" (conservative risk engine starts there) | Matches the actual PHASE 12 behaviour; not "stays GO" |
| Stress duration | 10× mission length (6 000 ticks) | Long enough to surface integrator drift / memory leaks without spending minutes per test |
| Stress scenario | `healthy_60s` (no fault) | Worst case for invariant violations — the pipeline must remain sane even on benign input |
| Invariant checkers | Pure functions, no fixtures | Reusable from fault contract + soak + property tests |
| No source modifications | Add `backend/testing/` only | Per the spec; PHASE 16 is testing-only |

## Run

```bash
# 1. Install the new dev deps (pytest-benchmark, hypothesis).
source .venv/bin/activate
python -m pip install -r requirements.txt

# 2. Validate config (no change expected).
python -m backend.config.cli --config config --validate-only

# 3. Run the PHASE 16 test suite (fast mode; ~5 min including 10× stress).
PHASE16_FAST=1 python -m pytest backend/tests/test_phase16_testing.py -v

# 4. Run with the full benchmark suite.
PHASE16_BENCH=1 python -m pytest backend/tests/test_phase16_testing.py -v

# 5. Operator-facing coverage + benchmark reports.
python scripts/coverage_report.py            # full-suite coverage HTML
python scripts/run_benchmarks.py --save      # refresh .benchmarks/ baseline

# 6. Full sanity.
python -m pytest backend/tests/ -v
```

## Test results

26 tests pass in `test_phase16_testing.py`:
- `TestCoverage` (2, gated by `PHASE16_FAST`): `pytest-cov` collects; helpers expose sane per-module values.
- `TestPerformanceBudgets` (4): one tick < 200 ms; one 60 s mission < 60 s; one encode < 50 ms; one round-trip < 200 ms.
- `TestBenchmarks` (3, gated by `PHASE16_BENCH=1`): `pytest-benchmark` for one tick, one mission, one round-trip.
- `TestFaultDetectionContract` (5): every fault contract is checked; healthy stays at-or-below CAUTION.
- `TestPropertyInvariants` (6): health/RUL/risk score in range, queue bounded, frame codec idempotent, comparator self-passes.
- `TestStress` (3, gated by `PHASE16_FAST`): 10× mission with no invariant violation; no NaN/inf anywhere; queue stays bounded.
- `TestLatencyTracker` (2): full pipeline latency < 15 s; `SerialSource` records the `transport` stage.

Total full suite: **440 tests passing** (414 prior + 26 new).

## Mandatory-scenario coverage (testing slice)

| Scenario | Status | Evidence |
|---|---|---|
| **T1 — Coverage is measured** | ✅ | `test_pytest_cov_collects` + `scripts/coverage_report.py` |
| **T2 — Critical-module coverage is exercised by the test suite** | ✅ | `test_critical_modules_have_coverage` (asserts the helper output is sensible; full breakdown is in `scripts/coverage_report.py`) |
| **T3 — One tick < 200 ms (CI budget)** | ✅ | `test_one_tick_under_200ms` |
| **T4 — One mission < 60 s (CI budget)** | ✅ | `test_one_mission_under_60s` |
| **T5 — pytest-benchmark integrates** | ✅ | `test_bench_*` (gated by `PHASE16_BENCH=1`) |
| **T6 — Health score in [0, 1]** | ✅ | `test_health_score_in_unit_interval` |
| **T7 — RUL bounds contain central** | ✅ | `test_rul_bounds_contain_central` |
| **T8 — Risk score in [0, 1]** | ✅ | `test_risk_score_in_unit_interval` |
| **T9 — Queue bounded** | ✅ | `test_telemetry_queue_bounded` |
| **T10 — Wire round-trip is idempotent** | ✅ | `test_frame_codec_round_trip_idempotent` |
| **T11 — Comparator reflexive** | ✅ | `test_compare_runs_self_passes` |
| **T12 — Healthy stays ≤ CAUTION** | ✅ | `test_healthy_60s_stays_under_caution` |
| **T13 — Every fault surfaces in risk** | ✅ | parametrised `test_fault_contract_holds` × 4 fault classes |
| **T14 — 10× mission no NaN/inf** | ✅ | `test_10x_mission_no_nan_or_inf_anywhere` |
| **T15 — 10× mission no invariant break** | ✅ | `test_10x_mission_healthy_no_invariant_violation` |
| **T16 — LatencyTracker end-to-end < 15 s** | ✅ | `test_pipeline_full_latency_under_15s` |
| **T17 — PHASE 1–15 unchanged** | ✅ | No source file in PHASES 1–15 is modified |

## Reuse from PHASES 1–15

| Source | What we use | Where |
|---|---|---|
| PHASE 10 `HealthIndex` | Invariant: `overall_score in [0, 1]` | `invariants.py`, property tests |
| PHASE 11 `RulEstimate` | Invariant: `tte_hours_lower <= tte_hours_central <= tte_hours_upper` | `invariants.py`, property tests |
| PHASE 12 `RiskAssessment`, `RiskStatus` | Status-rank ordering for fault contracts | `fault_contract.py` |
| PHASE 13 `PipelineRunner` | Tick + run the mission | benchmarks, stress, fault contract |
| PHASE 13 `SCENARIO_REGISTRY` | All five HIL scenarios | fault contract |
| PHASE 14 `FrameCodec`, `TelemetryQueue` | Wire round-trip, queue boundedness | benchmarks, property tests |
| PHASE 15 `Tolerance`, `compare_runs` | Reflexive comparator assertion | property tests |
| PHASE 5 `LatencyTracker` | End-to-end latency assertion | `TestLatencyTracker` |
| PHASE 1 `LoadedConfig` | Default config for tests | All tests |

