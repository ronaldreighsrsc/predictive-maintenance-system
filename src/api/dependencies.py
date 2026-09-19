"""
Dependencias de FastAPI para inyección del motor StagedTriageEngine.
"""

from fastapi import Request
from src.models.staged_triage_engine import StagedTriageEngine

# Singleton por defecto si la app no ha instanciado uno en app.state
_default_engine = None


def get_triage_engine(request: Request) -> StagedTriageEngine:
    global _default_engine
    if hasattr(request.app.state, "triage_engine") and request.app.state.triage_engine is not None:
        return request.app.state.triage_engine

    if _default_engine is None:
        _default_engine = StagedTriageEngine()
    return _default_engine
