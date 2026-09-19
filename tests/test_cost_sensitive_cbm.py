import pytest
from src.models.cost_sensitive_cbm import CostSensitiveCBMDecider, IndustrialCostMatrix


def test_analytical_threshold_calculation():
    matrix = IndustrialCostMatrix(cost_false_positive=8000.0, cost_false_negative=350000.0)
    decider = CostSensitiveCBMDecider(matrix)

    expected_theta = 8000.0 / (350000.0 + 8000.0)
    assert decider.optimal_critical_threshold == pytest.approx(expected_theta, rel=1e-5)
    assert decider.optimal_critical_threshold == pytest.approx(0.022346, rel=1e-3)


def test_bayesian_decision_overcomes_argmax_blindness():
    """
    Caso de estudio minero:
    P(Normal) = 0.85, P(Warning) = 0.10, P(Critical) = 0.05 (5%)
    Argmax tradicional elige 'Normal' (85%).
    Decisor Bayesiano detecta que 5% > 2.23% y despacha inmediatamente a taller,
    evitando una rotura de motor de $350,000 USD.
    """
    decider = CostSensitiveCBMDecider()
    probs = [0.85, 0.10, 0.05]
    decision = decider.decide_maintenance_action(probs)

    assert decision["urgency"] == "CRITICAL"
    assert decision["action"] == "DESPACHO_INMEDIATO_TALLER"
    assert decision["traditional_argmax_decision"] == "NORMAL"
    assert decision["economic_divergence"] is True
    assert decision["avoided_risk_usd"] == 338000.0


def test_warning_decision():
    decider = CostSensitiveCBMDecider()
    # P(Critical) es menor al 2.23%, pero P(Warning) es 25% (>= 20%)
    probs = [0.74, 0.25, 0.01]
    decision = decider.decide_maintenance_action(probs)

    assert decision["urgency"] == "WARNING"
    assert decision["action"] == "INSPECCION_SIGUIENTE_CAMBIO_TURNO"
    assert decision["traditional_argmax_decision"] == "NORMAL"
    assert decision["economic_divergence"] is True


def test_normal_operation():
    decider = CostSensitiveCBMDecider()
    probs = [0.95, 0.04, 0.01]
    decision = decider.decide_maintenance_action(probs)

    assert decision["urgency"] == "NORMAL"
    assert decision["action"] == "OPERACION_CONTINUA_NORMAL"
    assert decision["avoided_risk_usd"] == 0.0
