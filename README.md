# Unofficial Rotax 912 Simulation & Streamlit Dashboard

[![CI Pipeline](https://github.com/dikshantkumar02/uav-digital-twin-/actions/workflows/ci.yml/badge.svg)](https://github.com/dikshantkumar02/uav-digital-twin-/actions)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

> ### ⚠️ IMPORTANT SAFETY DISCLAIMER
> 1. **This is an unofficial simulation-only project.**
> 2. **It is not an official BRP-Rotax product or certified engine model.**
> 3. **It must NOT be used for real aircraft operation, flight-critical control, aircraft certification, maintenance release, or real engine limit determination.**
> 4. **Performance maps (BSFC, EGT, CHT) are synthetic numerical approximations and are NOT manufacturer-provided data.**
> 5. **The API and dashboard strictly provide read-only simulation telemetry and must never be connected to real engine actuators, throttles, ignition systems, or flight controls.**

---

## Overview

This repository provides an easy-to-deploy, modular, and secure physics-based simulation of the **Rotax 912 S/ULS (100 hp)** aircraft engine, consisting of:
- **FastAPI Backend (`app/` / `main.py`)**: High-performance asynchronous API serving steady-state thermodynamic & performance calculations, parameter validation, and health/readiness endpoints.
- **Streamlit Dashboard (`dashboard/` / `dashboard.py`)**: Interactive operator interface with real-time parameter sweeps, responsive gauges, exceedance alerts, and live backend health monitoring.
- **Container Infrastructure (`Dockerfile`, `Dockerfile.dashboard`, `docker-compose.yml`)**: Production-ready, non-root multi-container architecture with native Python healthchecks and private networking.

---

## Target Project Structure

```text
project-root/
├── app/
│   ├── __init__.py                # Package definition and metadata
│   ├── main.py                    # FastAPI application instance, CORS, error handling
│   ├── models.py                  # Pydantic v2 validation models and stable schemas
│   ├── engine.py                  # Bilinear map interpolator & Rotax 912 physics simulator
│   ├── config.py                  # Typed environment settings loader
│   └── routes/
│       ├── __init__.py
│       └── simulation.py          # POST /simulate endpoint definition
├── dashboard/
│   └── dashboard.py               # Streamlit application with error handling & metrics
├── config/
│   └── rotax_912_simulation.yaml  # Rotax 912 geometry, limits, and synthetic maps
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest path bootstrap
│   ├── test_health.py             # Root, /health, /ready endpoint tests
│   └── test_simulation.py         # Physics, limits, and Pydantic validation tests
├── .streamlit/
│   └── config.toml                # Headless server & dark aeronautical theme
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules
├── .dockerignore                  # Docker build context exclusions
├── Dockerfile                     # Production FastAPI backend container (non-root)
├── Dockerfile.dashboard           # Production Streamlit dashboard container (non-root)
├── docker-compose.yml             # Orchestrated multi-service stack with health checks
├── requirements.txt               # Pure runtime dependencies
├── requirements-dev.txt           # Test, lint, and development dependencies
├── README.md                      # Architecture and operational documentation
├── Makefile                       # Development task automation
└── .github/
    └── workflows/
        └── ci.yml                 # Automated linting, test matrix, and Docker build checks
```

---

## Configuration & Environment Variables

All settings are configured using environment variables or a local `.env` file (copy from `.env.example`). True secrets use `CHANGE_ME` as placeholder.

| Variable Name | Default Value | Description |
|:---|:---|:---|
| `APP_ENV` | `development` | Environment mode (`development`, `staging`, `production`). |
| `PORT` | `8000` | Port on which the FastAPI backend listens. |
| `DASHBOARD_PORT` | `8501` | Port on which the Streamlit dashboard listens. |
| `BACKEND_URL` | `http://localhost:8000` | Target URL used by Streamlit to reach FastAPI (`http://backend:8000` in Docker). |
| `ENABLE_API_DOCS` | `true` | Exposes Swagger UI (`/docs`) and ReDoc (`/redoc`) when `true`. Set to `false` in production. |
| `ALLOWED_ORIGINS` | `http://localhost:8501` | Comma-delimited list of allowed CORS origins for FastAPI. |
| `ENGINE_CONFIG_PATH` | `config/rotax_912_simulation.yaml` | Path to engine specifications and synthetic performance maps. |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## Quickstart: Local Development

### 1. Prerequisites
- Python 3.11 or 3.12
- Git

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/dikshantkumar02/uav-digital-twin-.git
cd uav-digital-twin-

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Copy environment template
cp .env.example .env

# Install dependencies
pip install -r requirements-dev.txt
```

### 3. Run FastAPI Backend
```bash
uvicorn main:app --reload --port 8000
```
- API root: `http://localhost:8000/`
- Health check: `http://localhost:8000/health`
- Readiness check: `http://localhost:8000/ready`
- Interactive API docs (if `ENABLE_API_DOCS=true`): `http://localhost:8000/docs`

### 4. Run Streamlit Dashboard
In a separate terminal:
```bash
streamlit run dashboard.py
```
- Dashboard UI: `http://localhost:8501`

---

## Deployment with Docker & Docker Compose

The project includes production-ready Docker configurations. Containers execute under an unprivileged `appuser` (UID 1000) with native HTTP health checks.

### Run Multi-Container Stack
```bash
# Build and start services in the background
docker compose up --build -d

# Verify container status and health
docker compose ps

# View real-time logs
docker compose logs -f

# Shut down services
docker compose down
```

Inside `docker-compose.yml`, the dashboard communicates with the backend via internal bridge networking (`http://backend:8000`) and will not start until the backend passes its readiness health check.

---

## API Endpoints Specification

### 1. `GET /`
Returns service metadata and mandatory safety disclaimer.
```json
{
  "service": "Unofficial Rotax 912 Simulation API",
  "version": "1.0.0",
  "status": "running",
  "docs_enabled": true,
  "safety_disclaimer": "SAFETY DISCLAIMER: This is an unofficial simulation-only project...",
  "endpoints": {
    "root": "/",
    "health": "/health",
    "ready": "/ready",
    "simulate": "/simulate",
    "docs": "/docs"
  }
}
```

### 2. `GET /health`
Liveness probe returning HTTP 200 with server status.

### 3. `GET /ready`
Readiness probe verifying engine configuration file loading and 2D map compiler status.

### 4. `POST /simulate`
Executes read-only physical simulation.

#### Sample Request:
```bash
curl -X POST "http://localhost:8000/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "rpm": 5200.0,
    "throttle": 0.85,
    "atmospheric_pressure_inhg": 29.92,
    "ambient_temperature_c": 15.0
  }'
```

#### Sample Response:
```json
{
  "crankshaft_rpm": 5200.0,
  "propeller_rpm": 2141.2,
  "estimated_power_kw": 61.2,
  "estimated_power_hp": 82.1,
  "manifold_pressure_inhg": 27.2,
  "fuel_flow_lph": 23.1,
  "bsfc_g_per_kwh": 271.5,
  "egt_c": 798.2,
  "cht_c": 116.4,
  "oil_pressure_psi": 50.0,
  "oil_temperature_c": 73.2,
  "warnings": [
    "NOTICE: Synthetic engine performance map used. Not certified manufacturer data."
  ],
  "model_metadata": {
    "model_id": "ROTAX-912S-SIM",
    "selected_variant": "Rotax 912 S/ULS (Carbureted, 100 hp)",
    "model_status": "UNOFFICIAL_SIMULATION",
    "calibration_status": "SYNTHETIC_RESEARCH_CALIBRATION",
    "manufacturer_validation": false,
    "certification_status": "UNNOTIFIED_NON_CERTIFIED",
    "reduction_gear_ratio": 2.4286
  },
  "safety_disclaimer": "SAFETY DISCLAIMER: Unofficial simulation-only model..."
}
```

#### Validation & Exceedance Logic:
- Returns **HTTP 422** if RPM is negative, exceeds simulation limit (6500 RPM), throttle is outside `[0.0, 1.0]`, values are non-finite, or naturally aspirated MAP exceeds ambient atmospheric pressure.
- Adds structured **warnings** if RPM exceeds maximum continuous rating (5500 RPM), take-off redline (5800 RPM), or if operating temperatures exceed safe limits.

---

## Testing & Quality Assurance

Automated unit tests validate schema validation, physical logic, error handling, and exceedance warnings:

```bash
# Run tests
make test
# Or directly:
pytest tests/ -v

# Run linting
make lint
# Or directly:
ruff check .
```

---

## License & Safety Notice

This software is released for academic, research, and simulation purposes only. Not affiliated with, endorsed by, or certified by BRP-Rotax GmbH & Co KG.

---

# Deployment

The application is structured for free-tier cloud deployment:
- **Frontend** → Netlify (Static Hosting)
- **Backend** → Render (Free Web Service)
- **Repository** → GitHub

---

## 1. Backend: Render Free Web Service

1. Connect your GitHub repository to [Render](https://render.com/).
2. Create a new **Web Service**.
3. Configure the service settings:
   - **Name**: `uav-digital-twin-backend` (or any preferred name)
   - **Language / Environment**: `Python 3`
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`
4. Set Environment Variables:
   - `PYTHON_VERSION`: `3.11.9`
   - `ALLOWED_ORIGINS`: `*` (or your Netlify site URL, e.g. `https://your-site.netlify.app`)
5. Health Check Path:
   - Set health check path to `/health`.
6. Once deployed, note your Render URL: `https://your-backend.onrender.com`.

---

## 2. Frontend: Netlify

1. Connect your GitHub repository to [Netlify](https://www.netlify.com/).
2. Netlify will automatically detect [`netlify.toml`](file:///c:/Users/DIKSHANT%20KUMAR/Desktop/prototype_3/netlify.toml):
   - **Base directory**: (leave blank / root)
   - **Build command**: (leave blank, no build step required)
   - **Publish directory**: `frontend`
3. Configure Backend Connection:
   - Option A (**URL Parameter - Instant Test**): Open your Netlify site with `?backend=https://your-backend.onrender.com`:
     ```text
     https://your-site.netlify.app/?backend=https://your-backend.onrender.com
     ```
   - Option B (**Static Config**): In [`frontend/assets/config.js`](file:///c:/Users/DIKSHANT%20KUMAR/Desktop/prototype_3/frontend/assets/config.js), set:
     ```javascript
     window.BACKEND_API_URL = "https://your-backend.onrender.com";
     window.BACKEND_WS_URL  = "wss://your-backend.onrender.com";
     ```
4. Deploy the site.

---

## 3. WebSocket Deployment Notes

- The real-time telemetry stream is served on `/api/stream`.
- Render natively supports persistent WebSockets over standard HTTPS/WSS on port 443 without special reverse proxy configuration.
- The frontend client automatically connects to `wss://your-backend.onrender.com/api/stream` when `BACKEND_API_URL` or `BACKEND_WS_URL` is configured. If WebSocket is closed or unsupported, the client automatically falls back to HTTP REST polling (`/api/snapshot/latest`).