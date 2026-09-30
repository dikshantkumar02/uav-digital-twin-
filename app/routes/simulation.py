"""Simulation endpoints for Rotax 912 engine model."""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.engine import Rotax912Simulator
from app.models import SimulationRequest, SimulationResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["simulation"])


def get_simulator(request: Request) -> Rotax912Simulator:
    """Dependency to retrieve the initialized simulator from app state."""
    sim = getattr(request.app.state, "simulator", None)
    if sim is None or not sim.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine simulation model is not initialized or ready.",
        )
    return sim


@router.post(
    "/simulate",
    response_model=SimulationResponse,
    summary="Run Rotax 912 physical simulation",
    description=(
        "Simulates Rotax 912 steady-state parameters (power, manifold pressure, temperatures, fuel flow) "
        "given operating conditions. Strictly unofficial simulation-only model."
    ),
    responses={
        200: {"description": "Simulation completed successfully."},
        422: {"description": "Validation error in simulation parameters."},
        503: {"description": "Engine simulation model not ready."},
    },
)
async def simulate(
    payload: SimulationRequest,
    simulator: Rotax912Simulator = Depends(get_simulator),
) -> SimulationResponse:
    """Compute engine operating point for given inputs."""
    try:
        return simulator.simulate(payload)
    except Exception as exc:
        log.exception("Simulation execution failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Simulation calculation error occurred.",
        ) from exc
