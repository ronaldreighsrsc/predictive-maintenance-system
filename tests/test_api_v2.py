import pytest
from fastapi.testclient import TestClient
from fastapii import app

client = TestClient(app)


def test_api_root_and_health():
    res_root = client.get("/")
    assert res_root.status_code == 200

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert "models_loaded" in res_health.json()


def test_telemetry_ingest_endpoint():
    payload = {
        "equipment_id": "CAEX-104",
        "engine_temp": 85.5,
        "oil_pressure": 45.1,
        "vibration_level": 2.52,
        "rpm": 1810.0,
        "fuel_consumption": 35.2,
        "coolant_temp": 78.3,
        "hydraulic_pressure": 3205.0,
        "operating_hours": 1250.0,
        "fleet_model": "KOMATSU 930E-4SE",
        "force_deep_path": False
    }
    response = client.post("/api/v2/telemetry/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["equipment_id"] == "CAEX-104"
    assert "triage" in data
    assert "cbm_decision" in data
    assert "optimal_critical_threshold" in data["cbm_decision"]


def test_buffer_status_and_reset_endpoints():
    eq_id = "CAEX-105"
    payload = {
        "equipment_id": eq_id,
        "engine_temp": 85.0,
        "oil_pressure": 45.0,
        "vibration_level": 2.5,
        "rpm": 1800.0,
        "fuel_consumption": 35.0,
        "coolant_temp": 78.0,
        "hydraulic_pressure": 3200.0,
        "operating_hours": 800.0
    }
    client.post("/api/v2/telemetry/ingest", json=payload)

    # Consultar estado del buffer
    res_buf = client.get(f"/api/v2/telemetry/buffer/{eq_id}")
    assert res_buf.status_code == 200
    assert res_buf.json()["equipment_id"] == eq_id
    assert res_buf.json()["buffer_length"] >= 1

    # Resetear buffer
    res_reset = client.post(f"/api/v2/telemetry/buffer/{eq_id}/reset")
    assert res_reset.status_code == 200
    assert res_reset.json()["reset_success"] is True


def test_maintenance_work_order_endpoint():
    payload = {
        "equipment_id": "CAEX-104",
        "engine_temp": 107.0,  # Sobrecalentamiento
        "oil_pressure": 44.0,
        "vibration_level": 2.5,
        "rpm": 1800.0,
        "fuel_consumption": 36.0,
        "coolant_temp": 97.0,
        "hydraulic_pressure": 3200.0,
        "health_score": 24.0,
        "fleet_model": "KOMATSU 930E-4SE"
    }
    response = client.post("/api/v2/maintenance/work-order", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["priority"] == 1
    assert data["sae_j1939_spn"] == 110
    assert "sap_pm_formatted_card" in data
    assert len(data["required_parts"]) > 0


def test_j1939_standards_catalog():
    response = client.get("/api/v2/maintenance/standards/j1939")
    assert response.status_code == 200
    data = response.json()
    assert "supported_subsystems" in data
    assert "COOLING_CIRCUIT" in data["supported_subsystems"]


def test_fleet_status_endpoint():
    response = client.get("/api/v2/fleet/status")
    assert response.status_code == 200
    data = response.json()
    assert "cbm_critical_threshold" in data
    assert data["cbm_critical_percentage"] == pytest.approx(2.23, rel=1e-1)
