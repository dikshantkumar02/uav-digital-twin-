"""
FastAPI app (PHASE 13 + PHASE 17).

The dashboard is a single FastAPI app that:

* serves a static HTML dashboard at ``/``;
* exposes REST endpoints for the latest snapshot, the snapshot
  history ring buffer, and the scenario registry;
* streams live snapshots over a WebSocket at ``/api/stream``;
* and is runnable in-process via :func:`create_app` + FastAPI's
  ``TestClient`` (no real socket binding required for tests).

A single :class:`~backend.dashboard.runner.PipelineRunner` is
shared across all endpoints. The pump task on app startup seeds
the queue with a few snapshots so a late-loading page is never
empty.

PHASE 17 (real-time streaming pipeline) adds:

* ``GET /api/latency`` — the actual measured per-stage latency
  plus the queue depths between stages. The endpoint reports
  the real numbers from the :class:`LatencyTracker` shared
  between the source, the preprocessor, and the per-tick
  pipeline; the dashboard shows the *real* latency, not a
  hard-coded constant.
* ``GET /api/replay/speeds`` — the available replay speeds.
* ``POST /api/replay/start`` — start a replay into the
  pipeline (in-memory only, returns a handle).
* ``POST /api/replay/stop`` — stop the active replay.
* ``GET /api/replay/stats`` — current replay stats.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.config import DashboardConfig, LoadedConfig
from backend.telemetry import LatencyTracker

from .replay import ReplayController, ReplaySpeed, ReplayStats
from .runner import DEFAULT_DT_S, PipelineRunner
from .scenarios import (
    DEFAULT_SCENARIO_NAME,
    SCENARIO_REGISTRY,
    ScenarioSpec,
    get_scenario,
)

log = logging.getLogger(__name__)

# Repo root for static file resolution.
REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------
class ScenarioRequest(BaseModel):
    name: str = Field(..., min_length=1)


class ReplayStartRequest(BaseModel):
    wire_path: str = Field(..., min_length=1)
    speed: str = Field("1x")


# ---------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------
def create_app(
    cfg: LoadedConfig,
    dashboard_cfg: DashboardConfig,
    *,
    initial_scenario: Optional[ScenarioSpec] = None,
    static_dir: Optional[Path] = None,
) -> FastAPI:
    """Create a FastAPI app wired to a single :class:`PipelineRunner`.

    Parameters
    ----------
    cfg:
        The full loaded configuration.
    dashboard_cfg:
        The dashboard's per-server configuration.
    initial_scenario:
        Optional override for the default scenario. Defaults to
        ``SCENARIO_REGISTRY[dashboard_cfg.default_scenario]`` (or
        :data:`DEFAULT_SCENARIO_NAME` if that key is missing).
    static_dir:
        Path to the static frontend directory. Defaults to
        ``<repo>/frontend``. If the directory does not contain an
        ``index.html``, ``GET /`` returns a 404 with a JSON error
        (instead of crashing the server).
    """
    if initial_scenario is None:
        initial_scenario = get_scenario(dashboard_cfg.default_scenario) or SCENARIO_REGISTRY[DEFAULT_SCENARIO_NAME]
    # PHASE 17: shared LatencyTracker so the runner's per-stage
    # timings are visible to /api/latency.
    latency_tracker = LatencyTracker()
    runner = PipelineRunner(
        cfg, initial_scenario,
        history_size=dashboard_cfg.history_size,
        dt_s=DEFAULT_DT_S,
        latency_tracker=latency_tracker,
    )
    # Run one synchronous tick at construction time so the very first
    # /api/snapshot/latest request is non-empty even before the
    # lifespan startup has finished.
    try:
        runner.tick()
    except Exception:  # noqa: BLE001
        pass
    static_root = static_dir or (REPO_ROOT / dashboard_cfg.static_dir)
    static_index = static_root / "index.html"

    # Per-app state.
    state: Dict[str, Any] = {
        "queue": asyncio.Queue(maxsize=dashboard_cfg.history_size),
        "ws_clients": set(),
        "pump_task": None,
        # PHASE 17: latency + replay state
        "latency_tracker": latency_tracker,
        "replay_task": None,
        "replay_controller": None,        # type: ignore[assignment]
        "replay_stats": ReplayStats(),
    }

    async def _seed_queue() -> None:
        """Run a few ticks at startup so a late-loading page is never empty."""
        for _ in range(5):
            try:
                snap = runner.tick()
            except Exception as exc:  # noqa: BLE001
                log.warning("startup tick failed: %s", exc)
                break
            try:
                state["queue"].put_nowait(snap)
            except asyncio.QueueFull:
                try:
                    state["queue"].get_nowait()
                except asyncio.QueueEmpty:
                    pass
                await state["queue"].put(snap)

    async def _pump_loop() -> None:
        """Advance the pipeline at the configured tick rate and fan out."""
        period = 1.0 / max(0.1, float(dashboard_cfg.tick_rate_hz))
        while True:
            try:
                snap = runner.tick()
            except Exception as exc:  # noqa: BLE001
                log.warning("tick failed; resetting: %s", exc)
                runner.reset()
                await asyncio.sleep(period)
                continue
            try:
                state["queue"].put_nowait(snap)
            except asyncio.QueueFull:
                # Drop the oldest so the latest is always delivered.
                try:
                    state["queue"].get_nowait()
                except asyncio.QueueEmpty:
                    pass
                await state["queue"].put(snap)
            # Reset when the mission is fully consumed so the dashboard
            # is "always on" (cycles through the same scenario).
            if runner.is_finished:
                runner.reset()
            await asyncio.sleep(period)

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        # Startup.
        await _seed_queue()
        state["pump_task"] = asyncio.create_task(_pump_loop())
        try:
            yield
        finally:
            # Shutdown.
            task = state.get("pump_task")
            if task is not None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
                state["pump_task"] = None

    app = FastAPI(title="Uav Piston Digital Twin Dashboard", lifespan=_lifespan)

    # ------------------------------------------------------------------
    # Static + REST endpoints
    # ------------------------------------------------------------------
    # Serve the static assets (CSS, JS, sub-directories like
    # /assets, /charts, /components, /pages, /services) from the
    # configured frontend directory.  Without this mount the
    # index.html references to /assets/dashboard.css return 404
    # and the page renders unstyled.
    if static_root.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(static_root / "assets")),
            name="frontend-assets",
        )
        # Mount the other sub-directories the frontend may pull
        # from (charts, components, pages, services).  Each is
        # mounted only if it actually exists on disk so a
        # minimal frontend (assets/ + index.html only) still works.
        for sub in ("charts", "components", "pages", "services"):
            sub_path = static_root / sub
            if sub_path.is_dir():
                app.mount(
                    f"/{sub}",
                    StaticFiles(directory=str(sub_path)),
                    name=f"frontend-{sub}",
                )

    @app.get("/")
    async def index() -> Any:
        if not static_index.exists():
            return JSONResponse(
                {
                    "error": "frontend index.html not found",
                    "expected_path": str(static_index),
                },
                status_code=404,
            )
        return FileResponse(str(static_index), media_type="text/html")

    @app.get("/api/snapshot/latest")
    async def snapshot_latest() -> Dict[str, Any]:
        snap = runner.last_snapshot
        if snap is None:
            return {"snapshot": None, "scenario_name": runner.scenario.name}
        return {"snapshot": snap.to_dict(), "scenario_name": runner.scenario.name}

    @app.get("/api/history")
    async def snapshot_history(limit: int = 60) -> Dict[str, Any]:
        if limit < 1:
            raise HTTPException(status_code=400, detail="limit must be >= 1")
        cap = min(int(limit), runner.history.maxlen if runner.history.maxlen else int(limit))
        history = list(runner.history)[-cap:]
        return {"snapshots": [s.to_dict() for s in history]}

    # PHASE 22 — unified diagnostic state endpoints. The wire
    # format is the same ``DiagnosticState.to_dict()`` schema
    # the dashboard, replay, and external consumers all read.
    @app.get("/api/diagnostic/latest")
    async def diagnostic_latest(rich: int = 0) -> Dict[str, Any]:
        diag = runner.latest_diagnostic
        if diag is None:
            return {
                "diagnostic": None,
                "scenario_name": runner.scenario.name,
            }
        payload = diag.to_rich_dict() if int(rich) else diag.to_dict()
        return {
            "diagnostic": payload,
            "scenario_name": runner.scenario.name,
        }

    @app.get("/api/diagnostic/history")
    async def diagnostic_history(limit: int = 60) -> Dict[str, Any]:
        if limit < 1:
            raise HTTPException(status_code=400, detail="limit must be >= 1")
        cap = min(int(limit), runner.diagnostic_history.maxlen or int(limit))
        history = list(runner.diagnostic_history)[-cap:]
        return {"diagnostics": [d.to_dict() for d in history]}

    # PHASE 23 — mission reliability view endpoints. Wire format
    # is exactly ``MissionReliabilityAssessment.to_dict()``: a
    # band, a confidence, and an explanation with a headline
    # and a list of named drivers. There is intentionally no
    # mission-success-probability number and no control commands
    # anywhere in the payload.
    @app.get("/api/reliability/latest")
    async def reliability_latest() -> Dict[str, Any]:
        rel = runner.latest_reliability
        if rel is None:
            return {
                "assessment": None,
                "scenario_name": runner.scenario.name,
            }
        return {
            "assessment": rel.to_dict(),
            "scenario_name": runner.scenario.name,
        }

    @app.get("/api/reliability/history")
    async def reliability_history(limit: int = 60) -> Dict[str, Any]:
        if limit < 1:
            raise HTTPException(status_code=400, detail="limit must be >= 1")
        cap = min(int(limit), runner.reliability_history.maxlen or int(limit))
        history = list(runner.reliability_history)[-cap:]
        return {"assessments": [r.to_dict() for r in history]}

    @app.get("/api/scenario")
    async def scenario_get() -> Dict[str, Any]:
        return {
            "name": runner.scenario.name,
            "spec": runner.scenario.to_dict(),
        }

    @app.get("/api/scenarios")
    async def scenarios_list() -> Dict[str, Any]:
        return {"scenarios": runner.list_scenarios()}

    @app.post("/api/scenario")
    async def scenario_set(body: ScenarioRequest) -> Dict[str, Any]:
        spec = get_scenario(body.name)
        if spec is None:
            raise HTTPException(
                status_code=404, detail=f"unknown scenario: {body.name}"
            )
        runner.set_scenario(spec)
        # Seed a few ticks so the next /api/snapshot/latest is non-null.
        for _ in range(2):
            try:
                runner.tick()
            except Exception:  # noqa: BLE001
                break
        return {"name": spec.name, "spec": spec.to_dict()}

    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "scenario": runner.scenario.name,
            "tick_count": runner.tick_count,
            "is_finished": runner.is_finished,
            "ws_clients": len(state["ws_clients"]),
        }

    # ------------------------------------------------------------------
    # PHASE 17 — measured latency + replay endpoints
    # ------------------------------------------------------------------
    @app.get("/api/latency")
    async def latency() -> Dict[str, Any]:
        """Return measured per-stage latency and queue depths.

        The numbers are the real values the
        :class:`LatencyTracker` has recorded across the running
        pipeline. The dashboard renders them as a real-time
        per-stage chart — not a constant.
        """
        tr: LatencyTracker = state["latency_tracker"]
        stages: Dict[str, Dict[str, float]] = {}
        total_samples = 0
        total_e2e = 0.0
        for name, stage in tr.summary().items():
            stages[name] = {
                "samples": int(stage.samples),
                "mean_s": float(stage.mean_s),
                "max_s": float(stage.max_s),
                "last_s": float(stage.last_s),
            }
            total_samples = max(total_samples, int(stage.samples))
            total_e2e += float(stage.last_s)
        # No queue-depth introspection for the synchronous path
        # (only the AsyncStreamingPipeline has queues).
        queue_depths: Dict[str, int] = {
            "in": 0,
            "validation": 0,
            "preprocessing": 0,
            "digital_twin": 0,
            "ai": 0,
            "risk": 0,
        }
        # If an async pipeline is attached, query its depths.
        pipeline = state.get("async_pipeline")
        if pipeline is not None:
            queue_depths = pipeline.queue_depths()
        return {
            "stages": stages,
            "end_to_end": {
                "samples": int(total_samples),
                "last_s": float(total_e2e),
            },
            "queue_depths": queue_depths,
        }

    @app.get("/api/replay/speeds")
    async def replay_speeds() -> Dict[str, Any]:
        return {"speeds": [s.value for s in ReplaySpeed]}

    @app.post("/api/replay/start")
    async def replay_start(body: ReplayStartRequest) -> Dict[str, Any]:
        pipeline = state.get("async_pipeline")
        if pipeline is None:
            raise HTTPException(
                status_code=400,
                detail="async pipeline not attached; cannot start replay",
            )
        # If a replay is already running, stop it first.
        existing = state.get("replay_task")
        if existing is not None and not existing.done():
            state["replay_controller"].stop()
            try:
                await existing
            except Exception:  # noqa: BLE001
                pass
        try:
            speed = ReplaySpeed(body.speed)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"unknown speed: {body.speed}; "
                       f"valid: {[s.value for s in ReplaySpeed]}",
            )
        wire_path = Path(body.wire_path)
        if not wire_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"wire.bin not found: {wire_path}",
            )
        controller = ReplayController(wire_path, pipeline, speed=speed)
        state["replay_controller"] = controller
        state["replay_stats"] = ReplayStats(speed=speed)
        state["replay_task"] = asyncio.create_task(controller.run())
        return {"speed": speed.value, "wire_path": str(wire_path)}

    @app.post("/api/replay/stop")
    async def replay_stop() -> Dict[str, Any]:
        controller: Optional[ReplayController] = state.get("replay_controller")
        if controller is None:
            return {"stopped": False, "reason": "no replay running"}
        controller.stop()
        task = state.get("replay_task")
        if task is not None:
            try:
                await task
            except Exception:  # noqa: BLE001
                pass
        return {"stopped": True, "stats": controller.stats.to_dict()}

    @app.get("/api/replay/stats")
    async def replay_stats() -> Dict[str, Any]:
        controller: Optional[ReplayController] = state.get("replay_controller")
        if controller is None:
            return {"running": False, "stats": state["replay_stats"].to_dict()}
        return {"running": True, "stats": controller.stats.to_dict()}

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------
    @app.websocket("/api/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        # Replay the last N snapshots so the page is instantly populated.
        for snap in list(runner.history)[-30:]:
            try:
                await ws.send_json(snap.to_dict())
            except Exception:  # noqa: BLE001
                break
        clients: Set[WebSocket] = state["ws_clients"]
        clients.add(ws)
        try:
            while True:
                snap = await state["queue"].get()
                await ws.send_json(snap.to_dict())
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            log.debug("ws error: %s", exc)
        finally:
            clients.discard(ws)

    # PHASE 22 — diagnostic stream: the same wire format the
    # /api/diagnostic/latest endpoint serves, pushed live.
    # Frontends that want only the unified state can subscribe
    # here without subscribing to the full snapshot stream.
    @app.websocket("/api/stream/diagnostic")
    async def stream_diagnostic(ws: WebSocket) -> None:
        await ws.accept()
        # Replay the last N diagnostic states so the page is
        # instantly populated.
        for diag in list(runner.diagnostic_history)[-30:]:
            try:
                await ws.send_json(diag.to_dict())
            except Exception:  # noqa: BLE001
                break
        try:
            while True:
                snap = await state["queue"].get()
                # The runner has already produced a fresh
                # DiagnosticState on the same tick; surface it.
                diag = runner.latest_diagnostic
                if diag is None:
                    continue
                await ws.send_json(diag.to_dict())
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            log.debug("ws error (diagnostic): %s", exc)

    return app


__all__ = ["create_app", "REPO_ROOT"]
