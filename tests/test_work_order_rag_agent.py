import pytest
from src.maintenance.work_order_rag_agent import MaintenanceWorkOrderAgent


def test_cooling_root_cause_diagnosis():
    agent = MaintenanceWorkOrderAgent()
    readings = {
        'engine_temp': 106.0,  # Muy alto (normal ~85)
        'oil_pressure': 44.0,
        'vibration_level': 2.5,
        'rpm': 1800.0,
        'fuel_consumption': 36.0,
        'coolant_temp': 98.0,  # Muy alto (normal ~78)
        'hydraulic_pressure': 3200.0
    }
    subsystem, z_scores = agent.diagnose_root_cause(readings)
    assert subsystem == "COOLING_CIRCUIT"
    assert z_scores['engine_temp'] > 4.0


def test_lubrication_root_cause_diagnosis():
    agent = MaintenanceWorkOrderAgent()
    readings = {
        'engine_temp': 86.0,
        'oil_pressure': 18.0,  # Críticamente bajo (normal ~45)
        'vibration_level': 2.5,
        'rpm': 1800.0,
        'fuel_consumption': 35.0,
        'coolant_temp': 78.0,
        'hydraulic_pressure': 3200.0
    }
    subsystem, z_scores = agent.diagnose_root_cause(readings)
    assert subsystem == "LUBRICATION_CIRCUIT"
    assert z_scores['oil_pressure'] < -3.0


def test_work_order_generation_structure():
    agent = MaintenanceWorkOrderAgent()
    readings = {
        'engine_temp': 105.0,
        'oil_pressure': 44.0,
        'vibration_level': 2.5,
        'rpm': 1800.0,
        'fuel_consumption': 36.5,
        'coolant_temp': 96.0,
        'hydraulic_pressure': 3200.0
    }
    cbm_decision = {
        "urgency": "CRITICAL",
        "action": "DESPACHO_INMEDIATO_TALLER",
        "p_critical": 0.08
    }
    weibull_rul = {
        "rul_p10_conservative": 5.4,
        "rul_p50_median": 12.0
    }

    wo = agent.generate_work_order(
        equipment_id="CAEX-104",
        sensor_readings=readings,
        health_score=22.5,
        cbm_decision=cbm_decision,
        weibull_rul=weibull_rul
    )

    assert wo["equipment_id"] == "CAEX-104"
    assert wo["priority"] == 1
    assert wo["sae_j1939_spn"] == 110
    assert wo["sae_j1939_fmi"] == 0
    assert len(wo["required_parts"]) >= 2
    assert len(wo["prescriptive_action_plan"]) >= 4
    assert "ORDEN DE TRABAJO AUTOMATIZADA" in wo["sap_pm_formatted_card"]
    assert "SPN 110" in wo["sap_pm_formatted_card"]
