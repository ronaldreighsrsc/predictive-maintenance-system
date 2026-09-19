import pytest
from src.evaluation.weibull_rul_estimator import WeibullRULEstimator, WeibullParameters


def test_weibull_confidence_intervals_ordering():
    estimator = WeibullRULEstimator()
    res = estimator.estimate_rul(health_score=85.0, operating_hours=200.0)

    p10 = res["rul_p10_conservative"]
    p50 = res["rul_p50_median"]
    p90 = res["rul_p90_optimistic"]

    # Propiedad matemática estocástica fundamental: P10 <= P50 <= P90
    assert p10 <= p50 <= p90
    assert res["uncertainty_spread"] == pytest.approx(p90 - p10, rel=1e-3)
    assert res["risk_tier"] == "OPERATIONAL_STABILITY"


def test_weibull_critical_health_drop():
    estimator = WeibullRULEstimator()
    # Health Score crítico (ej. 15%)
    res = estimator.estimate_rul(health_score=15.0, operating_hours=1200.0)

    assert res["rul_p10_conservative"] < 30.0
    assert res["risk_tier"] in ["CRITICAL_MAINTENANCE_WINDOW", "NEAR_TERM_ATTENTION"]


def test_weibull_monotonicity():
    estimator = WeibullRULEstimator()
    res_high = estimator.estimate_rul(health_score=95.0)
    res_low = estimator.estimate_rul(health_score=35.0)

    # A menor health score, menor vida remanente
    assert res_high["rul_p50_median"] > res_low["rul_p50_median"]
    assert res_high["rul_p10_conservative"] > res_low["rul_p10_conservative"]
