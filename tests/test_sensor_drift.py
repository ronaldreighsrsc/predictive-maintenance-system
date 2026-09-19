import pytest
import numpy as np
from src.monitoring.sensor_drift_detector import SensorDriftDetector


def test_normal_consistent_telemetry():
    detector = SensorDriftDetector()
    normal_reading = {
        'engine_temp': 85.0,
        'oil_pressure': 45.0,
        'vibration_level': 2.5,
        'rpm': 1800.0,
        'fuel_consumption': 35.0,
        'coolant_temp': 78.0,
        'hydraulic_pressure': 3200.0
    }
    res = detector.push_and_audit_sensors("CAEX-101", normal_reading)
    assert res["is_sensor_fault"] is False
    assert res["classification"] == "HEALTHY_TELEMETRY"
    assert len(res["discrepancies"]) == 0


def test_thermodynamic_cross_sensor_discrepancy():
    """
    Simula una anomalía física donde el sensor de motor marca 108°C
    pero el refrigerante está frío (60°C) y las RPM en 1000.
    El detector debe clasificarlo como SENSOR_CALIBRATION_DRIFT.
    """
    detector = SensorDriftDetector()
    sensor_fault_reading = {
        'engine_temp': 108.0,
        'oil_pressure': 45.0,
        'vibration_level': 2.5,
        'rpm': 1000.0,
        'fuel_consumption': 28.0,
        'coolant_temp': 60.0,  # Imposible que motor esté a 108C con coolant a 60C en baja carga
        'hydraulic_pressure': 3200.0
    }
    res = detector.push_and_audit_sensors("CAEX-102", sensor_fault_reading)
    assert res["is_sensor_fault"] is True
    assert res["classification"] == "SENSOR_CALIBRATION_DRIFT"
    assert any("Termistor de motor" in d for d in res["discrepancies"])


def test_kolmogorov_smirnov_drift_detection():
    detector = SensorDriftDetector(history_window=20)
    eq_id = "CAEX-103"

    # Enviar lecturas con un sensor de vibración artificialmente desplazado (offset +3.0 mm/s sobre baseline 2.5)
    for i in range(25):
        reading = {
            'engine_temp': 85.0 + np.random.normal(0, 1),
            'oil_pressure': 45.0 + np.random.normal(0, 1),
            'vibration_level': 5.5 + np.random.normal(0, 0.2),  # Desviado a 5.5 mm/s (anomalía de deriva)
            'rpm': 1800.0 + np.random.normal(0, 20),
            'fuel_consumption': 35.0 + np.random.normal(0, 0.5),
            'coolant_temp': 78.0 + np.random.normal(0, 1),
            'hydraulic_pressure': 3200.0 + np.random.normal(0, 20)
        }
        res = detector.push_and_audit_sensors(eq_id, reading)

    # Debe haber ejecutado KS-test y detectado deriva
    assert "vibration_level" in res["ks_test_results"]
    assert res["ks_test_results"]["vibration_level"]["has_distribution_drift"] is True
