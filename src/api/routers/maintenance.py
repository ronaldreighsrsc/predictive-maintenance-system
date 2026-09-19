"""
Router de Gestión de Mantenimiento y Agente RAG SAP PM (/api/v2/maintenance).
Genera órdenes de trabajo estructuradas con códigos de falla SAE J1939 y normas ISO 13374.
"""

from fastapi import APIRouter, Depends
from src.api.schemas import WorkOrderManualRequest, WorkOrderOutput
from src.api.dependencies import get_triage_engine
from src.models.staged_triage_engine import StagedTriageEngine

router = APIRouter(prefix="/api/v2/maintenance", tags=["Maintenance & SAP PM Work Orders"])


@router.post("/work-order", response_model=WorkOrderOutput)
def create_prescriptive_work_order(
    request: WorkOrderManualRequest,
    engine: StagedTriageEngine = Depends(get_triage_engine)
):
    """
    Genera bajo demanda una Orden de Trabajo SAP PM para un equipo basada en lecturas de sensores.
    Realiza atribución de causa raíz y recuperación de repuestos OEM.
    """
    sensor_dict = {
        'engine_temp': request.engine_temp,
        'oil_pressure': request.oil_pressure,
        'vibration_level': request.vibration_level,
        'rpm': request.rpm,
        'fuel_consumption': request.fuel_consumption,
        'coolant_temp': request.coolant_temp,
        'hydraulic_pressure': request.hydraulic_pressure
    }

    cbm_decision = {
        "urgency": "CRITICAL" if request.health_score < 40 else "WARNING",
        "action": "DESPACHO_INMEDIATO_TALLER" if request.health_score < 40 else "INSPECCION_SIGUIENTE_CAMBIO_TURNO",
        "p_critical": 0.15 if request.health_score < 40 else 0.05
    }

    weibull_rul = engine.rul_estimator.estimate_rul(health_score=request.health_score)

    wo = engine.work_order_agent.generate_work_order(
        equipment_id=request.equipment_id,
        sensor_readings=sensor_dict,
        health_score=request.health_score,
        cbm_decision=cbm_decision,
        weibull_rul=weibull_rul,
        fleet_model=request.fleet_model or "KOMATSU 930E-4SE"
    )

    return wo


@router.get("/standards/j1939")
def list_supported_j1939_standards(
    engine: StagedTriageEngine = Depends(get_triage_engine)
):
    """Retorna el catálogo indexado de códigos de falla SAE J1939 y catálogos OEM soportados."""
    kb = engine.work_order_agent.KNOWLEDGE_BASE
    catalog = {}
    for sub, dtc in kb.items():
        catalog[sub] = {
            "spn": dtc.spn,
            "fmi": dtc.fmi,
            "system": dtc.system,
            "description": dtc.description,
            "parts_count": len(dtc.oem_parts),
            "prescriptive_steps_count": len(dtc.prescriptive_steps)
        }
    return {
        "standard": "SAE J1939 Heavy Duty Vehicle Applications",
        "supported_subsystems": catalog
    }
