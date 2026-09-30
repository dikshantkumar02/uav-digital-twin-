"""
Streamlit interactive dashboard for the Rotax 912 engine simulation.

SAFETY DISCLAIMER:
1. This is an unofficial simulation-only project.
2. It is not an official BRP-Rotax product or certified engine model.
3. It must not be used for real aircraft operation, flight-critical control,
   aircraft certification, maintenance release, or real engine limit determination.
4. Never claim that synthetic engine maps are manufacturer-provided data.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple
import requests
import streamlit as st

# Environment-driven backend URL
BACKEND_URL_DEFAULT = os.getenv("BACKEND_URL", "http://localhost:8000")

SAFETY_DISCLAIMER_TEXT = (
    "⚠️ **IMPORTANT SAFETY DISCLAIMER**: This application is an **unofficial simulation-only prototype**. "
    "It is NOT certified by BRP-Rotax, FAA, EASA, or DGCA. "
    "It must NEVER be used for real flight operations, flight-critical control, aircraft maintenance release, "
    "or engine limits determination. Performance maps are synthetic approximations."
)

st.set_page_config(
    page_title="Rotax 912 Simulation Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Prominent Safety Disclaimer Banner
st.error(SAFETY_DISCLAIMER_TEXT)

st.title("Rotax 912 Engine Digital Simulation")
st.caption("Unofficial Aero-Piston Propulsion Predictive State & Parameter Visualizer")

# Sidebar Controls
st.sidebar.header("⚙️ Simulation Configuration")
st.sidebar.info("Simulation inputs are evaluated on the FastAPI backend.")

backend_url = st.sidebar.text_input(
    "FastAPI Backend URL",
    value=BACKEND_URL_DEFAULT,
    help="URL where the simulation FastAPI backend is listening (e.g. http://localhost:8000).",
)

st.sidebar.subheader("Engine Operational Controls")

crankshaft_rpm = st.sidebar.slider(
    "Crankshaft RPM",
    min_value=1000.0,
    max_value=6200.0,
    value=5000.0,
    step=50.0,
    help="Engine crankshaft speed. Rotax 912 rated continuous: 5500 RPM, Take-off limit: 5800 RPM.",
)

throttle = st.sidebar.slider(
    "Throttle Position (0.0 - 1.0)",
    min_value=0.0,
    max_value=1.0,
    value=0.85,
    step=0.05,
    help="Throttle position from idle (0.0) to full throttle (1.0).",
)

use_custom_map = st.sidebar.checkbox(
    "Override Manifold Pressure (MAP)",
    value=False,
    help="Explicitly command manifold absolute pressure instead of throttle-derived transfer.",
)

ambient_press = st.sidebar.number_input(
    "Atmospheric Pressure (inHg)",
    min_value=15.0,
    max_value=32.0,
    value=29.92,
    step=0.1,
    help="Ambient atmospheric pressure (ISA sea level: 29.92 inHg / 101.325 kPa).",
)

custom_map = None
if use_custom_map:
    custom_map = st.sidebar.number_input(
        "Manifold Pressure (inHg)",
        min_value=10.0,
        max_value=float(ambient_press),
        value=min(26.5, float(ambient_press)),
        step=0.5,
        help="Commanded manifold pressure. Naturally aspirated MAP cannot exceed ambient pressure.",
    )

ambient_temp = st.sidebar.number_input(
    "Outside Air Temperature (°C)",
    min_value=-40.0,
    max_value=55.0,
    value=15.0,
    step=1.0,
    help="Ambient air temperature for thermal equilibrium calculation.",
)

run_sim = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)


def check_backend_health(url: str) -> Dict[str, Any] | None:
    try:
        resp = requests.get(f"{url.rstrip('/')}/health", timeout=3.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def execute_simulation(
    payload: Dict[str, Any], url: str
) -> Tuple[Dict[str, Any] | None, str | None, int | None]:
    try:
        resp = requests.post(f"{url.rstrip('/')}/simulate", json=payload, timeout=5.0)
        status_code = resp.status_code
        try:
            data = resp.json()
        except Exception:
            return None, "Backend returned an invalid, non-JSON response payload.", status_code

        if status_code == 200:
            return data, None, status_code
        elif status_code == 422:
            detail = data.get("message") or data.get("detail") or "Validation Error (HTTP 422)"
            return None, f"Input Validation Error: {detail}", status_code
        elif status_code == 500:
            msg = data.get("message") or "Internal Backend Computation Failure"
            return None, f"Backend Internal Error (HTTP 500): {msg}", status_code
        elif status_code == 503:
            return None, "Backend Simulation Model Not Ready (HTTP 503)", status_code
        else:
            return None, f"Unexpected Backend Response HTTP {status_code}: {data}", status_code
    except requests.exceptions.ConnectionError:
        return None, f"Could not connect to FastAPI backend at {url}. Ensure server is running.", None
    except requests.exceptions.Timeout:
        return None, "Connection to simulation backend timed out (> 5.0s).", None
    except Exception as e:
        return None, f"Network communication error: {str(e)}", None


# Status Bar
health_info = check_backend_health(backend_url)
col_stat1, col_stat2, col_stat3 = st.columns([2, 2, 2])
with col_stat1:
    if health_info:
        st.success(f"Backend Online ({health_info.get('status', 'ok')})")
    else:
        st.warning("Backend Offline / Unreachable")

with col_stat2:
    st.info("Target: Rotax 912 S/ULS (100 hp)")

with col_stat3:
    st.info("Reduction Gear Ratio: 2.4286:1")

# Build simulation payload
payload = {
    "rpm": float(crankshaft_rpm),
    "throttle": float(throttle),
    "manifold_pressure_inhg": float(custom_map) if custom_map is not None else None,
    "atmospheric_pressure_inhg": float(ambient_press),
    "ambient_temperature_c": float(ambient_temp),
    "altitude_m": 0.0,
}

# Auto-execute on load or button press
sim_data, err_msg, status_code = execute_simulation(payload, backend_url)

if err_msg:
    st.error(f"❌ {err_msg}")
    st.info("💡 Ensure the FastAPI backend is running: `uvicorn main:app --port 8000` or via Docker.")
elif sim_data:
    # 1. Primary Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Crankshaft RPM",
            value=f"{sim_data['crankshaft_rpm']:.0f} RPM",
            delta="Max Cont: 5500 RPM",
            delta_color="off",
        )
    with m2:
        st.metric(
            label="Propeller RPM",
            value=f"{sim_data['propeller_rpm']:.0f} RPM",
            help="Propeller speed after 2.4286:1 reduction gearbox.",
        )
    with m3:
        st.metric(
            label="Estimated Power",
            value=f"{sim_data['estimated_power_hp']:.1f} HP",
            delta=f"{sim_data['estimated_power_kw']:.1f} kW",
        )
    with m4:
        st.metric(
            label="Manifold Pressure (MAP)",
            value=f"{sim_data['manifold_pressure_inhg']:.2f} inHg",
            help="Absolute manifold pressure.",
        )

    # 2. Secondary Metrics Row
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric(
            label="Fuel Flow",
            value=f"{sim_data['fuel_flow_lph']:.1f} L/h",
            help="Calculated from BSFC and power output.",
        )
    with s2:
        st.metric(
            label="BSFC",
            value=f"{sim_data['bsfc_g_per_kwh']:.0f} g/kWh",
            help="Brake Specific Fuel Consumption.",
        )
    with s3:
        st.metric(
            label="Exhaust Gas Temp (EGT)",
            value=f"{sim_data['egt_c']:.1f} °C",
            delta="Limit: 850 °C",
            delta_color="inverse",
        )
    with s4:
        st.metric(
            label="Cylinder Head Temp (CHT)",
            value=f"{sim_data['cht_c']:.1f} °C",
            delta="Limit: 135 °C",
            delta_color="inverse",
        )

    # 3. Lubrication Metrics Row
    l1, l2, l3, l4 = st.columns(4)
    with l1:
        st.metric(
            label="Oil Pressure",
            value=f"{sim_data['oil_pressure_psi']:.1f} PSI",
            help="Operating limits: 11.6 - 101.5 PSI.",
        )
    with l2:
        st.metric(
            label="Oil Temperature",
            value=f"{sim_data['oil_temperature_c']:.1f} °C",
            help="Operating limits: 50 - 130 °C.",
        )
    with l3:
        meta = sim_data.get("model_metadata", {})
        st.metric(label="Model Variant", value=meta.get("model_id", "ROTAX-912S-SIM"))
    with l4:
        st.metric(label="Certification", value=meta.get("certification_status", "NON_CERTIFIED"))

    # Warnings Section
    warnings = sim_data.get("warnings", [])
    if warnings:
        st.subheader("⚠️ Simulation Notices & Exceedances")
        for w in warnings:
            if "EXCEEDANCE" in w or "LIMIT" in w:
                st.error(w)
            elif "CAUTION" in w or "WARNING" in w:
                st.warning(w)
            else:
                st.info(w)

    # Model Metadata & Safety Notes Card
    with st.expander("ℹ️ Model Architecture & Provenance Details", expanded=False):
        meta = sim_data.get("model_metadata", {})
        st.json(meta)
        st.write(sim_data.get("safety_disclaimer", ""))

# Bottom Navigation
st.divider()
st.caption(
    "Rotax 912 Simulation Stack · FastAPI Backend: `/simulate` · Streamlit Dashboard · Ready for Docker Deployment"
)
