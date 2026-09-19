from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np

# Import models
from src.models.xgb_predictor import MaintenanceXGBoostPredictor
from src.models.isolation_forest import MaintenanceIsolationForest
from src.models.autoencoder_deep import MaintenanceDeepAutoencoder
from src.models.autoencoder_lstm import MaintenanceLSTMAutoencoder
from src.models.gan_detector import MaintenanceGANDetector
from src.models.staged_triage_engine import StagedTriageEngine
from src.api.routers import telemetry, maintenance, fleet

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Loading models into memory...")
    try:
        ml_models['xgb'] = MaintenanceXGBoostPredictor.load("./models/saved_models/xgb.pkl")
        ml_models['iso'] = MaintenanceIsolationForest.load("./models/saved_models/iso.pkl")
        ml_models['ae_deep'] = MaintenanceDeepAutoencoder.load("./models/saved_models/ae_deep.pkl")
        ml_models['ae_lstm'] = MaintenanceLSTMAutoencoder.load("./models/saved_models/ae_lstm.pkl")
        ml_models['gan'] = MaintenanceGANDetector.load("./models/saved_models/gan.pkl")
        print("✅ All models loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading models (Did you run main_training.py?): {e}")

    # Inicializar motor Staged Triage v2.0
    app.state.triage_engine = StagedTriageEngine(native_models=ml_models)
    
    yield
    ml_models.clear()

app = FastAPI(
    title="Predictive Maintenance API v2.0",
    description="Industrial-grade Condition-Based Maintenance (CBM / ISO 13374) for Heavy Mining Fleets",
    version="2.0.0",
    lifespan=lifespan
)

# Montar routers modulares v2.0
app.include_router(telemetry.router)
app.include_router(maintenance.router)
app.include_router(fleet.router)

class SensorReading(BaseModel):
    engine_temp: float
    oil_pressure: float
    vibration_level: float
    rpm: float
    fuel_consumption: float
    coolant_temp: float
    hydraulic_pressure: float
    operating_hours: float
    engine_temp_roll_mean_5: float
    engine_temp_roll_std_5: float
    engine_temp_roll_mean_10: float
    engine_temp_roll_std_10: float
    engine_temp_roll_mean_20: float
    engine_temp_roll_std_20: float
    oil_pressure_roll_mean_5: float
    oil_pressure_roll_std_5: float
    oil_pressure_roll_mean_10: float
    oil_pressure_roll_std_10: float
    oil_pressure_roll_mean_20: float
    oil_pressure_roll_std_20: float
    vibration_level_roll_mean_5: float
    vibration_level_roll_std_5: float
    vibration_level_roll_mean_10: float
    vibration_level_roll_std_10: float
    vibration_level_roll_mean_20: float
    vibration_level_roll_std_20: float
    rpm_roll_mean_5: float
    rpm_roll_std_5: float
    rpm_roll_mean_10: float
    rpm_roll_std_10: float
    rpm_roll_mean_20: float
    rpm_roll_std_20: float
    fuel_consumption_roll_mean_5: float
    fuel_consumption_roll_std_5: float
    fuel_consumption_roll_mean_10: float
    fuel_consumption_roll_std_10: float
    fuel_consumption_roll_mean_20: float
    fuel_consumption_roll_std_20: float
    coolant_temp_roll_mean_5: float
    coolant_temp_roll_std_5: float
    coolant_temp_roll_mean_10: float
    coolant_temp_roll_std_10: float
    coolant_temp_roll_mean_20: float
    coolant_temp_roll_std_20: float
    hydraulic_pressure_roll_mean_5: float
    hydraulic_pressure_roll_std_5: float
    hydraulic_pressure_roll_mean_10: float
    hydraulic_pressure_roll_std_10: float
    hydraulic_pressure_roll_mean_20: float
    hydraulic_pressure_roll_std_20: float
    engine_temp_delta: float
    oil_pressure_delta: float
    vibration_level_delta: float
    rpm_delta: float
    fuel_consumption_delta: float
    coolant_temp_delta: float
    hydraulic_pressure_delta: float
    engine_temp_ema_10: float
    oil_pressure_ema_10: float
    vibration_level_ema_10: float
    rpm_ema_10: float
    fuel_consumption_ema_10: float
    coolant_temp_ema_10: float
    hydraulic_pressure_ema_10: float
    temp_oil_ratio: float
    temp_coolant_diff: float
    vibration_per_rpm: float
    fuel_efficiency: float

    class Config:
        json_schema_extra = {
            "example": {
                "engine_temp": 85.0, "oil_pressure": 45.0, "vibration_level": 0.5,
                "rpm": 1800.0, "fuel_consumption": 12.0, "coolant_temp": 75.0,
                "hydraulic_pressure": 200.0, "operating_hours": 500.0,
                "engine_temp_roll_mean_5": 84.5, "engine_temp_roll_std_5": 1.2,
                "engine_temp_roll_mean_10": 84.0, "engine_temp_roll_std_10": 1.5,
                "engine_temp_roll_mean_20": 83.5, "engine_temp_roll_std_20": 1.8,
                "oil_pressure_roll_mean_5": 44.5, "oil_pressure_roll_std_5": 0.8,
                "oil_pressure_roll_mean_10": 44.0, "oil_pressure_roll_std_10": 1.0,
                "oil_pressure_roll_mean_20": 43.5, "oil_pressure_roll_std_20": 1.2,
                "vibration_level_roll_mean_5": 0.48, "vibration_level_roll_std_5": 0.05,
                "vibration_level_roll_mean_10": 0.47, "vibration_level_roll_std_10": 0.06,
                "vibration_level_roll_mean_20": 0.46, "vibration_level_roll_std_20": 0.07,
                "rpm_roll_mean_5": 1790.0, "rpm_roll_std_5": 20.0,
                "rpm_roll_mean_10": 1785.0, "rpm_roll_std_10": 25.0,
                "rpm_roll_mean_20": 1780.0, "rpm_roll_std_20": 30.0,
                "fuel_consumption_roll_mean_5": 11.8, "fuel_consumption_roll_std_5": 0.3,
                "fuel_consumption_roll_mean_10": 11.7, "fuel_consumption_roll_std_10": 0.4,
                "fuel_consumption_roll_mean_20": 11.6, "fuel_consumption_roll_std_20": 0.5,
                "coolant_temp_roll_mean_5": 74.5, "coolant_temp_roll_std_5": 1.0,
                "coolant_temp_roll_mean_10": 74.0, "coolant_temp_roll_std_10": 1.2,
                "coolant_temp_roll_mean_20": 73.5, "coolant_temp_roll_std_20": 1.5,
                "hydraulic_pressure_roll_mean_5": 199.0, "hydraulic_pressure_roll_std_5": 2.0,
                "hydraulic_pressure_roll_mean_10": 198.0, "hydraulic_pressure_roll_std_10": 3.0,
                "hydraulic_pressure_roll_mean_20": 197.0, "hydraulic_pressure_roll_std_20": 4.0,
                "engine_temp_delta": 0.5, "oil_pressure_delta": -0.2,
                "vibration_level_delta": 0.01, "rpm_delta": 10.0,
                "fuel_consumption_delta": 0.1, "coolant_temp_delta": 0.3,
                "hydraulic_pressure_delta": -1.0,
                "engine_temp_ema_10": 84.2, "oil_pressure_ema_10": 44.3,
                "vibration_level_ema_10": 0.48, "rpm_ema_10": 1792.0,
                "fuel_consumption_ema_10": 11.9, "coolant_temp_ema_10": 74.8,
                "hydraulic_pressure_ema_10": 199.5,
                "temp_oil_ratio": 1.89, "temp_coolant_diff": 10.0,
                "vibration_per_rpm": 0.28, "fuel_efficiency": 6.67
            }
        }

@app.get("/")
def read_root():
    return {"message": "Predictive Maintenance API is running! 🚀"}

@app.get("/health")
def health_check():
    loaded = list(ml_models.keys())
    return {"status": "ok", "models_loaded": loaded}

@app.post("/predict")
def predict_equipment_health(reading: SensorReading):
    data = reading.model_dump()
    features = np.array([[data[k] for k in data.keys()]])

    results = {}

    # XGBoost Multi-Class (Normal/Warning/Critical)
    if 'xgb' in ml_models:
        pred, probs = ml_models['xgb'].predict(features)
        state_map = {0: "Normal", 1: "Warning", 2: "Critical"}
        results['xgboost'] = {
            "predicted_state": state_map.get(int(pred[0]), "Unknown"),
            "probabilities": {state_map[i]: float(probs[0][i]) for i in range(len(probs[0]))}
        }

    # Isolation Forest
    if 'iso' in ml_models:
        pred, score = ml_models['iso'].predict(features)
        results['isolation_forest'] = {"is_anomaly": bool(pred[0]), "anomaly_score": float(score[0])}

    # Deep Autoencoder (Health Score)
    if 'ae_deep' in ml_models:
        pred, health = ml_models['ae_deep'].predict_anomaly(features)
        results['deep_autoencoder'] = {"is_anomaly": bool(pred[0]), "health_score": float(health[0])}

    # LSTM Autoencoder (Health Score)
    if 'ae_lstm' in ml_models:
        pred, health = ml_models['ae_lstm'].predict_anomaly(features)
        results['lstm_autoencoder'] = {"is_anomaly": bool(pred[0]), "health_score": float(health[0])}

    # GAN (Health Score)
    if 'gan' in ml_models:
        pred, health = ml_models['gan'].predict_anomaly(features)
        results['gan'] = {"is_anomaly": bool(pred[0]), "health_score": float(health[0])}

    return {
        "status": "Sensor reading analyzed",
        "predictions": results
    }
