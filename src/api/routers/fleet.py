"""
Router de Gestión de Flota y Confiabilidad Global (/api/v2/fleet).
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any, List

from src.api.dependencies import get_triage_engine
from src.models.staged_triage_engine import StagedTriageEngine

router = APIRouter(prefix="/api/v2/fleet", tags=["Fleet Reliability Overview"])


@router.get("/status")
def get_fleet_operational_status(
    engine: StagedTriageEngine = Depends(get_triage_engine)
):
    """Retorna el estado de monitoreo de la flota minera en memoria."""
    active_eq = engine.buffer.get_active_equipments()
    statuses = [engine.buffer.get_buffer_status(eq) for eq in active_eq]

    return {
        "active_monitored_trucks": len(active_eq),
        "equipment_list": active_eq,
        "cbm_critical_threshold": engine.cbm_decider.optimal_critical_threshold,
        "cbm_critical_percentage": round(engine.cbm_decider.optimal_critical_threshold * 100.0, 2),
        "cost_matrix_usd": {
            "false_positive": engine.cbm_decider.costs.cost_false_positive,
            "false_negative": engine.cbm_decider.costs.cost_false_negative,
            "true_positive": engine.cbm_decider.costs.cost_true_positive
        },
        "equipments_in_memory": statuses
    }
