"""
Testing + performance package (PHASE 16).

Public re-exports so callers can write::

    from backend.testing import (
        InvariantViolation,
        assert_health_invariants, assert_rul_invariants,
        assert_risk_invariants, assert_no_nan_inf,
        assert_snapshot_dict_invariants, assert_typed_snapshot_invariants,
        FaultContract, FAULT_CONTRACTS, check_fault_contract, FaultContractError,
        measure_coverage, module_coverage,
    )

This package is read-only with respect to PHASES 1–15 — it
imports the production code and asserts on it.
"""

from .coverage import measure_coverage, module_coverage
from .fault_contract import (
    FAULT_CONTRACTS,
    FaultContract,
    FaultContractError,
    check_fault_contract,
)
from .invariants import (
    InvariantViolation,
    assert_anomaly_invariants,
    assert_health_invariants,
    assert_no_nan_inf,
    assert_risk_invariants,
    assert_rul_invariants,
    assert_snapshot_dict_invariants,
    assert_typed_snapshot_invariants,
)

__all__ = [
    "FAULT_CONTRACTS",
    "FaultContract",
    "FaultContractError",
    "InvariantViolation",
    "assert_anomaly_invariants",
    "assert_health_invariants",
    "assert_no_nan_inf",
    "assert_risk_invariants",
    "assert_rul_invariants",
    "assert_snapshot_dict_invariants",
    "assert_typed_snapshot_invariants",
    "check_fault_contract",
    "measure_coverage",
    "module_coverage",
]
