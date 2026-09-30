"""FastAPI application deployment entry point for Render.

Exports `app` for:
    uvicorn app:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path so 'backend.*' imports succeed
# regardless of whether the working directory is the repo root or 'backend/'.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.middleware.cors import CORSMiddleware
from backend.config import load_config, load_dashboard_config
from backend.dashboard.app import create_app

# Load configuration (deterministic, reads config/*.yaml)
cfg = load_config()
dashboard_cfg = load_dashboard_config()

# If PORT is provided in environment (e.g., Render sets $PORT), override config
if "PORT" in os.environ:
    try:
        dashboard_cfg = dashboard_cfg.model_copy(update={"port": int(os.environ["PORT"])})
    except Exception:
        pass

# Instantiate the digital twin dashboard FastAPI app
app = create_app(cfg, dashboard_cfg)

# CORS middleware for cross-origin Netlify -> Render communication
frontend_origins_raw = os.getenv("ALLOWED_ORIGINS", os.getenv("FRONTEND_URL", "*"))
allowed_origins = [o.strip() for o in frontend_origins_raw.split(",") if o.strip()]

if not allowed_origins or "*" in allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Root health endpoint for platform health checks (Render /health)
# Leaves existing /api/health endpoint intact
@app.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "healthy",
        "service": "uav-digital-twin-backend",
        "scenario": getattr(dashboard_cfg, "default_scenario", "default"),
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", dashboard_cfg.port))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
