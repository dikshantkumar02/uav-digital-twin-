"""FastAPI backend application for Rotax 912 simulation.

SAFETY DISCLAIMER:
1. This is an unofficial simulation-only project.
2. It is not an official BRP-Rotax product or certified engine model.
3. It must not be used for real aircraft operation, flight-critical control,
   aircraft certification, maintenance release, or real engine limit determination.
4. Never claim that synthetic engine maps are manufacturer-provided data.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import Any, AsyncGenerator, Dict, List

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.engine import Rotax912Simulator
from app.models import HealthResponse, ReadyResponse, RootResponse
from app.routes.simulation import router as simulation_router

settings = get_settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("rotax912.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context to manage simulator lifecycle safely."""
    log.info("Starting Rotax 912 Simulation Backend in %s mode", settings.app_env)
    try:
        simulator = Rotax912Simulator(config_path=settings.engine_config_path)
        app.state.simulator = simulator
        log.info("Simulation engine ready: %s", simulator.metadata.model_id)
    except Exception as exc:
        log.error("Fatal error loading engine simulation model: %s", exc)
        app.state.simulator = None
    yield
    log.info("Shutting down Rotax 912 Simulation Backend.")


app = FastAPI(
    title="Unofficial Rotax 912 Simulation API",
    version="1.0.0",
    description=(
        "Unofficial physics-based simulation API for Rotax 912 engine series. "
        "Strictly for simulation, research, and educational purposes. "
        "NOT FOR REAL FLIGHT OR CERTIFICATION."
    ),
    docs_url="/docs" if settings.enable_api_docs else None,
    redoc_url="/redoc" if settings.enable_api_docs else None,
    openapi_url="/openapi.json" if settings.enable_api_docs else None,
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Format Pydantic validation errors cleanly with HTTP 422 without raw exceptions."""
    log.warning("Validation error on %s: %s", request.url.path, exc.errors())
    clean_errors: List[Dict[str, Any]] = []
    for err in exc.errors():
        item = {
            "type": err.get("type", "value_error"),
            "loc": [str(x) for x in err.get("loc", ())],
            "msg": str(err.get("msg", "")),
        }
        if "input" in err:
            try:
                item["input"] = jsonable_encoder(err["input"])
            except Exception:
                item["input"] = str(err["input"])
        clean_errors.append(item)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": clean_errors,
            "message": "Supplied simulation parameters failed schema validation.",
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Shield internal stack traces from leaking to clients in production."""
    log.exception("Unhandled internal exception during request to %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred during request processing.",
        },
    )


@app.get(
    "/",
    response_model=RootResponse,
    summary="Root metadata and safety disclaimer",
)
async def root() -> RootResponse:
    """Root metadata endpoint with official safety disclaimer."""
    return RootResponse(
        service="Unofficial Rotax 912 Simulation API",
        version="1.0.0",
        status="running",
        docs_enabled=settings.enable_api_docs,
        safety_disclaimer=(
            "SAFETY DISCLAIMER: This is an unofficial simulation-only project. "
            "It is not an official BRP-Rotax product or certified engine model. "
            "It must not be used for real aircraft operation, flight-critical control, "
            "aircraft certification, maintenance release, or real engine limit determination."
        ),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness health check",
)
async def health() -> HealthResponse:
    """Fast liveness check for container orchestration and uptime monitors."""
    return HealthResponse(
        status="healthy",
        app_env=settings.app_env,
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness check including model load status",
)
async def ready(request: Request) -> ReadyResponse:
    """Readiness probe verifying that engine simulation config and maps are loaded."""
    sim = getattr(request.app.state, "simulator", None)
    if sim is None or not sim.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine model is not ready.",
        )
    return ReadyResponse(
        status="ready",
        model_ready=True,
        model_id=sim.metadata.model_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


app.include_router(simulation_router)
