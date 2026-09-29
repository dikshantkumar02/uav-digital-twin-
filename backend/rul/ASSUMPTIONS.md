# RUL — Assumptions

This document consolidates the mathematical and statistical
assumptions behind the RUL calculation. Every assumption is
recorded so the operator can audit the estimate and understand
under what conditions the RUL number is meaningful — and when
it is not.

The RUL module's contract is "**no fake outputs**": when any of
these assumptions is violated, the estimate's `status` becomes
`RUL_UNCERTAIN` rather than the system emitting a confident
number. See `backend/rul/types.py` and `aggregate.py` for the
exact mapping.

---

## 1. Wear model

The current wear rate (hours⁻¹) is the product of four positive
multiplicative terms:

```
rate = base
     * (1 + k_health * (1 - health.overall_score))
     * (1 + k_fault  * sum(severity * prob for fault, prob
                            in contributing_faults))
     * (1 + k_trend  * (1 if trend == DEGRADING else 0))
     * (1 + wear)                         # wear accelerates wear
```

* `base` is the healthy engine's wear rate at rated power, read
  from `EngineConfig.degradation.base_wear_per_hour` (default
  `1e-5`).
* `k_health = 2.0`, `k_fault = 4.0`, `k_trend = 0.5`. These are
  transparent, deterministic, and documented here so the
  operator can adjust them via `ClosedFormWearRate(...)`.
* The model is **monotone in every input**: more health loss,
  more fault severity, more trend pressure, or more current wear
  ⇒ more wear rate. This is intentional — it makes the closed
  form interpretable. The trained RF may learn non-monotone
  corrections.

## 2. Trained-model fallback

When a `TrainedWearModel` is loaded, the calculator uses the
RF's `rate_per_hour(...)` instead of the closed-form. The
selector (`backend/rul/selector.py`) chooses between the two
based on training-data sufficiency and held-out MAE (see
`MIN_TRAIN_SCENARIOS = 50` and `MAX_RELATIVE_MAE = 1.10`).

## 3. Time-To-End-of-Life (TTE) — Monte Carlo

The MC layer (`backend/rul/monte_carlo.py`) propagates the
current wear forward under three stochastic terms:

* **Per-step multiplicative noise** on the rate, Gaussian with
  `sigma = 0.20`, clamped to `[-0.95, 5.0]` to avoid zero or
  negative rates.
* **Linear drift** in the rate over the horizon, parameterised
  by `fault_drift_per_hour = 1e-5`. The drift models the
  observation that wear typically accelerates with time.
* The current wear at each step is `current_wear + cumsum(rate *
  dt * drift * noise)`. The first step at which wear crosses
  `max_wear` (default 1.0) is the TTE for that sample.

`n_samples = 200` trajectories are run per tick. The central
TTE is the median; bounds are configurable percentiles
(defaults: 5th / 95th).

## 4. Horizon

`DEFAULT_HORIZON_HOURS = 100,000` (~11 years of continuous
operation). A healthy engine at the default base rate
(`1e-5 / h`) reaches `max_wear = 1.0` in `~100,000 h`. The
horizon is long enough that healthy engines do not cap at the
horizon, and the bounds remain meaningful.

Trajectories that don't reach EOL within the horizon are
counted in the `confidence` denominator (so `confidence`
reflects "fraction of MC samples that reached EOL within the
horizon") but are not included in the TTE distribution.

## 5. Trend

The RUL `trend` is the change in central TTE over the last
`TREND_WINDOW = 10` ticks, thresholded at
`TREND_EPSILON_HOURS = 0.5`:

* `IMPROVING` — central TTE increased by more than 0.5 h.
* `STABLE` — change within ±0.5 h.
* `DEGRADING` — central TTE decreased by more than 0.5 h.
* `INSUFFICIENT_DATA` — fewer than 10 ticks of history.

The trend is **independent of the health-index trend**: the
TTE may be stable even while the health index is degrading
(the health impact hasn't propagated to the wear rate yet).

## 6. Status thresholds

`RUL_UNCERTAIN` is returned when:
* `health_label == INSUFFICIENT_DATA`, **or**
* `tte.confidence < 0.20` (fewer than 20 % of MC samples
  reached EOL within the horizon).

`RUL_CRITICAL` is returned when:
* `health_label == CRITICAL`, **or**
* `0.20 <= tte.confidence < 0.40`.

`RUL_DEGRADED` is returned when:
* `health_label == DEGRADED`, **or**
* `health_trend == DEGRADING`.

Otherwise: `RUL_OK`.

The mapping is intentionally conservative — when in doubt,
return `RUL_UNCERTAIN`. The spec's "no fake outputs" rule
beats a falsely-confident number every time.

## 7. Inputs that may be missing

The calculator is robust to missing inputs:

* `state` is optional. Without it, the current wear comes from
  `health.wear`, which itself defaults to 0.0.
* `residual_trend` is optional. Without it, the new
  residual-trend feature columns default to 0.0; the model
  still produces an estimate (it just doesn't see the trend
  signal).
* `time_s` is optional. Without it, the timestamp is taken
  from `state.time_s` then `health.time_s`, then defaults to
  0.0.

In all of these cases, the resulting `RulEstimate` is
**valid** but may carry a `RUL_UNCERTAIN` status if the
missing input caused `health` itself to be insufficient.

## 8. NaN/inf in the residual trend

`ResidualTrendInput.__post_init__` replaces any `NaN`/`+inf`/
`-inf` in the z-score history with 0.0. A single bad residual
should not blow up the wear-rate model.

## 9. Training data: closed-form-as-truth

The synthetic training set in `build_dataset(...)` uses the
**closed-form** wear rate plus a small Gaussian noise term as
the ground truth. This is a deliberate choice: the closed-form
is the same function the calculator falls back to when no
trained model is loaded, so the RF learns its bias, not
something unrelated. The noise term (`5 %` standard
deviation) keeps the training set non-degenerate without
spending hours running slow simulator scenarios.

In production, the operator is expected to retrain the RF on
real telemetry once it is available; the sidecar `.version`
file detects schema mismatches and warns.

## 10. Feature schema versioning

The feature vector is versioned via `MODEL_VERSION =
"phase11-rul-1.1.0"`. Adding a column is a breaking change:
any RF trained on the old schema is loaded with a
`ModelVersionWarning` so the operator can decide whether the
saved artefact is still trustworthy. The schema change from
1.0 → 1.1 added the 5 residual-trend columns (see
`backend/rul/types.py::ResidualTrendInput`).

## 11. Evaluation

`backend/rul/evaluation.py` reports four standard metrics:

* **MAE** — mean absolute error in hours.
* **RMSE** — root mean squared error in hours.
* **MAPE** — mean absolute percentage error. Returned as
  `None` whenever any ground truth is ~0 (MAPE is undefined
  there). We do not report `inf` or `nan` here — that would
  be the "fabricate precision" anti-pattern.
* **PICP** — prediction-interval coverage probability, with
  `picp_target = 0.90` for the 5 / 95 % interval. PICP below
  target flags the bounds as under-confident; PICP above
  target flags them as over-confident. A `notes` field in
  `RulEvalReport` carries the verdict.

## 12. Plotting

`backend/rul/plotting.py` renders the four required charts as
**SVG** files. SVG is a text format that any browser /
dashboard / doc-renderer can display, and the renderer has
**no third-party dependency** (no matplotlib, no kaleido). The
charts are deterministic given the same inputs. The function
names are:

* `plot_health_vs_time(health_history, path)`
* `plot_rul_vs_time(rul_history, path)`
* `plot_actual_vs_predicted(eval_pairs, path)`
* `plot_prediction_uncertainty(eval_pairs, path)`

Each returns the path it wrote to.

---

**Bottom line.** The RUL number is a *transparent projection
under explicit assumptions*, not a black-box prediction. The
assumptions above are the contract between the model and the
operator: if any of them is wrong, the status field will tell
you, and the notes field will say why.
