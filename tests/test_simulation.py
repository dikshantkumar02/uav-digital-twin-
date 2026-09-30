"""Tests for Rotax 912 physical simulation endpoint and validations."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_simulation_nominal_point(client: TestClient):
    payload = {
        "rpm": 5000.0,
        "throttle": 0.8,
        "ambient_temperature_c": 15.0,
        "atmospheric_pressure_inhg": 29.92,
    }
    resp = client.post("/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["crankshaft_rpm"] == 5000.0
    assert data["propeller_rpm"] > 0
    assert data["estimated_power_kw"] > 0
    assert data["estimated_power_hp"] > 0
    assert data["manifold_pressure_inhg"] > 0
    assert data["fuel_flow_lph"] > 0
    assert data["bsfc_g_per_kwh"] > 0
    assert data["egt_c"] > 0
    assert data["cht_c"] > 0
    assert "SAFETY DISCLAIMER" in data["safety_disclaimer"]
    assert any("Synthetic" in w for w in data["warnings"])
    assert data["model_metadata"]["certification_status"] == "UNNOTIFIED_NON_CERTIFIED"


def test_simulation_continuous_rpm_warning(client: TestClient):
    # RPM 5600 is between rated 5500 and redline 5800
    payload = {
        "rpm": 5600.0,
        "throttle": 0.95,
        "ambient_temperature_c": 15.0,
        "atmospheric_pressure_inhg": 29.92,
    }
    resp = client.post("/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert any("Maximum Continuous Rating" in w for w in data["warnings"])


def test_simulation_redline_rpm_warning(client: TestClient):
    # RPM 5900 is above redline 5800
    payload = {
        "rpm": 5900.0,
        "throttle": 1.0,
        "ambient_temperature_c": 15.0,
        "atmospheric_pressure_inhg": 29.92,
    }
    resp = client.post("/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert any("Take-off Redline" in w for w in data["warnings"])


def test_validation_negative_rpm(client: TestClient):
    payload = {"rpm": -100.0, "throttle": 0.5}
    resp = client.post("/simulate", json=payload)
    assert resp.status_code == 422


def test_validation_rpm_exceeds_max(client: TestClient):
    payload = {"rpm": 7000.0, "throttle": 0.5}
    resp = client.post("/simulate", json=payload)
    assert resp.status_code == 422


def test_validation_invalid_throttle_bounds(client: TestClient):
    resp_low = client.post("/simulate", json={"rpm": 3000.0, "throttle": -0.1})
    assert resp_low.status_code == 422

    resp_high = client.post("/simulate", json={"rpm": 3000.0, "throttle": 1.1})
    assert resp_high.status_code == 422


def test_validation_naturally_aspirated_map_exceeds_ambient(client: TestClient):
    # Naturally aspirated MAP cannot exceed atmospheric pressure
    payload = {
        "rpm": 4500.0,
        "throttle": 0.7,
        "atmospheric_pressure_inhg": 29.0,
        "manifold_pressure_inhg": 31.5,
    }
    resp = client.post("/simulate", json=payload)
    assert resp.status_code == 422
    assert "cannot exceed ambient atmospheric pressure" in resp.text
