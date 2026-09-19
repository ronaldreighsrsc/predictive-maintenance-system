"""
Router de Ingesta y Streaming de Telemetría CAN Bus (/api/v2/telemetry).
Recibe únicamente las 7 lecturas crudas del camión y gestiona el buffer O(1).
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from src.api.schemas import RawTelemetryReading, TelemetryIngestResponse
from src.api.dependencies import get_triage_engine
from src.models.staged_triage_engine import StagedTriageEngine

router = APIRouter(prefix="/api/v2/telemetry", tags=["CAN Telemetry Streaming"])


@router.post("/ingest", response_model=TelemetryIngestResponse)
def ingest_telemetry_reading(
    reading: RawTelemetryReading,
    engine: StagedTriageEngine = Depends(get_triage_engine)
):
    """
    Ingesta en tiempo real del paquete crudo de 7 sensores del bus CAN del camión.
    Ejecuta en memoria el buffer O(1) (< 1.2 ms) y el triage en cascada (Fast-Path / Deep-Path).
    """
    raw_dict = {
        'engine_temp': reading.engine_temp,
        'oil_pressure': reading.oil_pressure,
        'vibration_level': reading.vibration_level,
        'rpm': reading.rpm,
        'fuel_consumption': reading.fuel_consumption,
        'coolant_temp': reading.coolant_temp,
        'hydraulic_pressure': reading.hydraulic_pressure
    }

    result = engine.process_raw_telemetry(
        equipment_id=reading.equipment_id,
        raw_sensors=raw_dict,
        operating_hours=reading.operating_hours or 0.0,
        fleet_model=reading.fleet_model or "KOMATSU 930E-4SE",
        force_deep_path=reading.force_deep_path or False
    )

    return result


@router.get("/buffer/{equipment_id}")
def get_equipment_buffer_status(
    equipment_id: str,
    engine: StagedTriageEngine = Depends(get_triage_engine)
):
    """Retorna el estado de calentamiento y tamaño del buffer circular en RAM para un camión."""
    status = engine.buffer.get_buffer_status(equipment_id)
    return status


@router.post("/buffer/{equipment_id}/reset")
def reset_equipment_buffer(
    equipment_id: str,
    engine: StagedTriageEngine = Depends(get_triage_engine)
):
    """Reinicia el buffer histórico de un equipo tras una intervención de mantenimiento."""
    success = engine.buffer.reset_equipment(equipment_id)
    return {
        "equipment_id": equipment_id,
        "reset_success": success,
        "message": f"Buffer para {equipment_id} reseteado exitosamente." if success else "Equipo no estaba en memoria."
    }
