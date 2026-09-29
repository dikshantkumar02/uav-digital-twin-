"""
RUL package — Remaining Useful Life + uncertainty (PHASE 11).

Public re-exports::

    from backend.rul import (
        # Core types
        RulCalculator, RulEstimate, RulStatus, RulTrend, RulBounds,
        RulModelStatus, ResidualTrendInput,
        # Wear-rate models
        ClosedFormWearRate, TrainedWearModel, WearRateDataset,
        # TTE simulation
        TteDistribution, simulate_tte,
        # Build + train + persist
        build_dataset, train_wear_model, save_model, load_model,
        MODEL_VERSION, ModelVersionWarning,
        # Feature columns
        FEATURE_NAMES, FEATURE_DIM, health_to_features,
        # Aggregator helpers
        aggregate, new_trend_history, update_trend_history,
        # Evaluation (MAE / RMSE / MAPE / PICP)
        RulEvalReport, evaluate_rul, mae, rmse, mape, picp,
        # Model selector (CF vs RF auto-pick)
        SelectionResult, select_model, MIN_TRAIN_SCENARIOS, MAX_RELATIVE_MAE,
        # Plotting (SVG, no extra deps)
        plot_health_vs_time, plot_rul_vs_time,
        plot_actual_vs_predicted, plot_prediction_uncertainty,
    )
"""

from .aggregate import aggregate, new_trend_history, update_trend_history
from .calculator import RulCalculator
from .evaluation import (
    DEFAULT_PICP_TARGET,
    RulEvalReport,
    evaluate_rul,
    mae,
    mape,
    picp,
    rmse,
)
from .model import (
    FEATURE_DIM,
    FEATURE_NAMES,
    MODEL_VERSION,
    ClosedFormWearRate,
    ModelVersionWarning,
    TrainedWearModel,
    WearRateDataset,
    build_dataset,
    health_to_features,
    load_model,
    load_rotax_rul_dataset,
    load_rotax_telemetry_features,
    save_model,
    train_wear_model,
)
from .monte_carlo import (
    DEFAULT_DT_S,
    DEFAULT_FAULT_DRIFT_PER_HOUR,
    DEFAULT_HORIZON_HOURS,
    DEFAULT_N_SAMPLES,
    DEFAULT_NOISE_STD,
    DEFAULT_QUANTILE_HIGH,
    DEFAULT_QUANTILE_LOW,
    TteDistribution,
    WearRateModel,
    simulate_tte,
)
from .plotting import (
    plot_actual_vs_predicted,
    plot_health_vs_time,
    plot_prediction_uncertainty,
    plot_rul_vs_time,
)
from .selector import (
    MAX_RELATIVE_MAE,
    MIN_TRAIN_SCENARIOS,
    SelectionResult,
    select_model,
)
from .types import (
    ResidualTrendInput,
    RulBounds,
    RulEstimate,
    RulModelStatus,
    RulStatus,
    RulTrend,
    TREND_EPSILON_HOURS,
    TREND_WINDOW,
    status_for,
    trend_for,
)

__all__ = [
    "DEFAULT_DT_S",
    "DEFAULT_FAULT_DRIFT_PER_HOUR",
    "DEFAULT_HORIZON_HOURS",
    "DEFAULT_N_SAMPLES",
    "DEFAULT_NOISE_STD",
    "DEFAULT_PICP_TARGET",
    "DEFAULT_QUANTILE_HIGH",
    "DEFAULT_QUANTILE_LOW",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "MAX_RELATIVE_MAE",
    "MIN_TRAIN_SCENARIOS",
    "MODEL_VERSION",
    "ModelVersionWarning",
    "ClosedFormWearRate",
    "ResidualTrendInput",
    "RulBounds",
    "RulCalculator",
    "RulEvalReport",
    "RulEstimate",
    "RulModelStatus",
    "RulStatus",
    "RulTrend",
    "TREND_EPSILON_HOURS",
    "TREND_WINDOW",
    "TrainedWearModel",
    "TteDistribution",
    "WearRateDataset",
    "WearRateModel",
    "aggregate",
    "build_dataset",
    "evaluate_rul",
    "health_to_features",
    "load_model",
    "load_rotax_rul_dataset",
    "load_rotax_telemetry_features",
    "mae",
    "mape",
    "new_trend_history",
    "picp",
    "plot_actual_vs_predicted",
    "plot_health_vs_time",
    "plot_prediction_uncertainty",
    "plot_rul_vs_time",
    "rmse",
    "save_model",
    "select_model",
    "simulate_tte",
    "status_for",
    "train_wear_model",
    "trend_for",
    "update_trend_history",
]
