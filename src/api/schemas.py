"""
Esquemas Pydantic v2 para la API REST v2 de Mantenimiento Predictivo Industrial.
Define contratos limpios y fuertemente tipados para telemetría CAN Bus, CBM y SAP PM.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class RawTelemetryReading(BaseModel):
    """Lectura cruda transmitida por el módem de telemetría CAN Bus del camión minero."""
    equipment_id: str = Field(..., description="ID del camión (ej. 'CAEX-104')", example="CAEX-104")
    engine_temp: float = Field(..., description="Temperatura del bloque de motor en °C", example=85.2)
    oil_pressure: float = Field(..., description="Presión de aceite de galería principal en PSI", example=44.8)
    vibration_level: float = Field(..., description="Nivel de vibración RMS en mm/s", example=2.52)
    rpm: float = Field(..., description="Revoluciones por minuto del motor", example=1810.0)
    fuel_consumption: float = Field(..., description="Tasa instantánea de consumo de combustible en L/h", example=35.1)
    coolant_temp: float = Field(..., description="Temperatura del líquido refrigerante en °C", example=78.1)
    hydraulic_pressure: float = Field(..., description="Presión del circuito hidráulico principal en PSI", example=3210.0)
    operating_hours: Optional[float] = Field(1200.0, description="Horas acumuladas de operación", example=1240.5)
    fleet_model: Optional[str] = Field("KOMATSU 930E-4SE", description="Modelo y fabricante del camión")
    force_deep_path: Optional[bool] = Field(False, description="Fuerza análisis temporal e informe profundo")

    class Config:
        json_schema_extra = {
            "example": {
                "equipment_id": "CAEX-104",
                "engine_temp": 86.1,
                "oil_pressure": 44.5,
                "vibration_level": 2.55,
                "rpm": 1805.0,
                "fuel_consumption": 35.2,
                "coolant_temp": 78.4,
                "hydraulic_pressure": 3205.0,
                "operating_hours": 1420.0,
                "fleet_model": "KOMATSU 930E-4SE",
                "force_deep_path": False
            }
        }


class TriageMetadata(BaseModel):
    stage: int = Field(..., description="Etapa ejecutada: 1 (Fast-Path) o 2 (Deep-Path)")
    mode: str = Field(..., description="FAST_PATH o DEEP_PATH_ACTIVATED")
    latency_ms: float = Field(..., description="Tiempo de inferencia en milisegundos")
    deep_path_triggered: bool = Field(..., description="Indica si se activó la etapa profunda")


class CBMDecisionOutput(BaseModel):
    action: str = Field(..., description="Acción de mantenimiento recomendada")
    urgency: str = Field(..., description="Nivel de urgencia: NORMAL, WARNING, CRITICAL")
    severity_code: int = Field(..., description="Código numérico de severidad: 0, 1, 2")
    probabilities: Dict[str, float] = Field(..., description="Probabilidades Normal/Warning/Critical")
    p_critical: float = Field(..., description="Probabilidad de falla crítica estimada")
    optimal_critical_threshold: float = Field(..., description="Umbral óptimo de Bayes (2.23%)")
    avoided_risk_usd: float = Field(..., description="Riesgo financiero de rotura evitado en USD")
    traditional_argmax_decision: str = Field(..., description="Decisión que habría tomado argmax simétrico")
    economic_divergence: bool = Field(..., description="Indica si la matriz de costo modificó la decisión")
    justification: str = Field(..., description="Explicación económica de la recomendación")


class WeibullRULOutput(BaseModel):
    health_score: float = Field(..., description="Score de salud continuo [0-100]")
    operating_hours: float = Field(..., description="Horómetro acumulado")
    rul_p10_conservative: float = Field(..., description="RUL P10: Estimación conservadora garantizada (90% certeza)")
    rul_p50_median: float = Field(..., description="RUL P50: Mediana esperada de ciclos hasta falla")
    rul_p90_optimistic: float = Field(..., description="RUL P90: Escenario optimista")
    uncertainty_spread: float = Field(..., description="Amplitud de incertidumbre (P90 - P10)")
    risk_tier: str = Field(..., description="Nivel de riesgo de programación")
    unit: str = Field("ciclos_operacionales", description="Unidad de medida")


class SensorAuditOutput(BaseModel):
    equipment_id: str
    classification: str = Field(..., description="HEALTHY_TELEMETRY, SENSOR_CALIBRATION_DRIFT, etc.")
    severity: str
    is_sensor_fault: bool = Field(..., description="True si la física indica sensor descalibrado, no motor roto")
    discrepancies: List[str] = Field(default_factory=list)
    recommendation: str


class WorkOrderOutput(BaseModel):
    work_order_id: str
    equipment_id: str
    fleet_model: str
    created_at: str
    priority: int
    priority_label: str
    subsystem: str
    subsystem_name: str
    sae_j1939_spn: int
    sae_j1939_fmi: int
    fault_code_label: str
    root_cause: str
    health_score: float
    rul_p10: float
    prescriptive_action_plan: List[str]
    required_parts: List[Dict[str, str]]
    sap_pm_formatted_card: str


class TelemetryIngestResponse(BaseModel):
    status: str
    equipment_id: str
    fleet_model: str
    triage: TriageMetadata
    health_score: float
    cbm_decision: CBMDecisionOutput
    weibull_rul: Optional[WeibullRULOutput] = None
    sensor_audit: Optional[SensorAuditOutput] = None
    work_order: Optional[WorkOrderOutput] = None
    buffer_status: Dict[str, Any]


class WorkOrderManualRequest(BaseModel):
    equipment_id: str
    engine_temp: float
    oil_pressure: float
    vibration_level: float
    rpm: float
    fuel_consumption: float
    coolant_temp: float
    hydraulic_pressure: float
    health_score: Optional[float] = 25.0
    fleet_model: Optional[str] = "KOMATSU 930E-4SE"
