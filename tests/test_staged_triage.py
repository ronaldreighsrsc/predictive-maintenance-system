import pytest
from src.models.staged_triage_engine import StagedTriageEngine
from src.models.xgb_predictor import MaintenanceXGBoostPredictor
from src.models.autoencoder_deep import MaintenanceDeepAutoencoder


def test_staged_triage_fast_path_and_deep_path():
    # Cargar modelos si están disponibles en disco
    models = {}
    try:
        models['xgb'] = MaintenanceXGBoostPredictor.load("./models/saved_models/xgb.pkl")
        models['ae_deep'] = MaintenanceDeepAutoencoder.load("./models/saved_models/ae_deep.pkl")
    except Exception:
        pass

    engine = StagedTriageEngine(native_models=models)

    # 1. Lectura normal saludable acorde a especificaciones de camión CAEX
    normal_reading = [85.0, 45.0, 2.5, 1800.0, 35.0, 78.0, 3200.0]
    
    # Warm up call (TensorFlow graph compilation on first inference)
    engine.process_raw_telemetry(
        equipment_id="CAEX-201",
        raw_sensors=normal_reading,
        operating_hours=1200.0
    )

    # Ingestar lecturas normales consecutivas para estabilizar el buffer
    for i in range(1, 20):
        engine.process_raw_telemetry(
            equipment_id="CAEX-201",
            raw_sensors=normal_reading,
            operating_hours=1200.0 + float(i)
        )

    res_fast = engine.process_raw_telemetry(
        equipment_id="CAEX-201",
        raw_sensors=normal_reading,
        operating_hours=1220.0
    )

    assert res_fast["status"] == "success"
    assert res_fast["equipment_id"] == "CAEX-201"
    assert "triage" in res_fast
    assert res_fast["triage"]["latency_ms"] < 250.0  # Latencia de procesamiento rápida en CPU

    # 2. Lectura crítica forzada con force_deep_path=True
    res_deep = engine.process_raw_telemetry(
        equipment_id="CAEX-202",
        raw_sensors=normal_reading,
        operating_hours=1200.0,
        force_deep_path=True
    )

    assert res_deep["triage"]["stage"] == 2
    assert res_deep["triage"]["mode"] == "DEEP_PATH_ACTIVATED"
    assert res_deep["weibull_rul"] is not None
    assert "rul_p10_conservative" in res_deep["weibull_rul"]
    assert res_deep["sensor_audit"] is not None
