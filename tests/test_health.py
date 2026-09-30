"""Tests for FastAPI backend root, health, and readiness endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "Unofficial Rotax 912 Simulation API"
    assert data["status"] == "running"
    assert "SAFETY DISCLAIMER" in data["safety_disclaimer"]
    assert "official BRP-Rotax product" in data["safety_disclaimer"]


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "app_env" in data


def test_ready_endpoint(client: TestClient):
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["model_ready"] is True
    assert "ROTAX" in data["model_id"]
