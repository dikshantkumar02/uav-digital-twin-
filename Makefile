# ==============================================================================
# Makefile for Unofficial Rotax 912 Simulation & Dashboard
# ==============================================================================

.PHONY: help install test lint format run-backend run-dashboard build up down clean

help:
	@echo "Available commands:"
	@echo "  make install         Install runtime and development dependencies"
	@echo "  make test            Run test suite with pytest"
	@echo "  make lint            Run ruff linter"
	@echo "  make format          Run ruff code formatter"
	@echo "  make run-backend     Run FastAPI development server (uvicorn main:app --reload)"
	@echo "  make run-dashboard   Run Streamlit dashboard (streamlit run dashboard.py)"
	@echo "  make build           Build Docker images"
	@echo "  make up              Start services with Docker Compose"
	@echo "  make down            Stop Docker Compose services"
	@echo "  make clean           Remove cached bytecode and test artifacts"

install:
	pip install --upgrade pip
	pip install -r requirements-dev.txt

test:
	python -m pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .

run-backend:
	uvicorn main:app --reload --port 8000

run-dashboard:
	streamlit run dashboard.py --server.port 8501

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"
	python -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.ruff_cache')]"
