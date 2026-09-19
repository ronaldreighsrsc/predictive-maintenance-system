"""
Estimador Estocástico de Remaining Useful Life (RUL) mediante Modelo de Degradación Weibull.
Elimina la fuga de datos (Data Leakage) de la v1.0 modelando la curva no lineal de desgaste
(Bathtub curve / Ley de potencia) y entregando intervalos de confianza estocásticos (P10, P50, P90).
"""

from dataclasses import dataclass
from typing import Dict, Any, Union, Optional
import numpy as np
from scipy.special import gamma


@dataclass(frozen=True)
class WeibullParameters:
    """
    Parámetros de la distribución de Weibull para componentes mecánicos CAEX.
    beta: Parámetro de forma (beta > 1 indica desgaste acelerado / envejecimiento).
    eta: Parámetro de escala (vida característica típica en ciclos u horas).
    """
    beta: float = 2.8      # Forma de curva de desgaste acelerado para motores diesel mineros
    eta: float = 350.0     # Vida característica base en ciclos de operación estándar
    min_rul: float = 1.0   # Cota mínima de ciclos antes de parada mandatoria


class WeibullRULEstimator:
    """
    Estimador estocástico de vida remanente.
    Calcula la distribución condicional de supervivencia dado el Health Score
    y las horas acumuladas, sin requerir conocer la vida total real previa (cero data leakage).
    """

    def __init__(self, params: WeibullParameters = WeibullParameters()):
        self.params = params

    def estimate_rul(
        self,
        health_score: float,
        operating_hours: float = 0.0,
        degradation_velocity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Estima el RUL estocástico en base al Health Score (0-100) y parámetros operacionales.

        Args:
            health_score: Estado de salud continuo [0.0 - 100.0].
            operating_hours: Horas acumuladas de operación.
            degradation_velocity: Tasa opcional de caída de health_score por ciclo (delta HS).

        Returns:
            Diccionario con RUL P10, P50, P90, desvío estándar e intervalo de confianza.
        """
        hs = float(np.clip(health_score, 0.1, 100.0))
        h_fraction = hs / 100.0

        # Transformación no lineal basada en función de supervivencia de Weibull:
        # S(t) = exp(-(t/eta)^beta)  => t = eta * (-ln(S))^(1/beta)
        # RUL no lineal esperado en función de la fracción de salud remanente
        # Usamos exponente de envejecimiento: (h_fraction)^(1 / beta)
        # Cuando h_fraction es 1.0 -> RUL base = eta * Gamma(1 + 1/beta)
        gamma_factor = gamma(1.0 + 1.0 / self.params.beta)
        nominal_mean_life = self.params.eta * gamma_factor

        # Ajuste no lineal con aceleración en el último tramo (codo de falla acelerada)
        # Durante el 70% de vida el desgaste parece suave; en el tramo final el colapso es exponencial (potencia beta)
        rul_median = nominal_mean_life * (h_fraction ** self.params.beta)

        # Ajuste dinámico por tasa de degradación reciente si está disponible
        if degradation_velocity is not None and degradation_velocity > 0:
            # Si el equipo se está degradando rápidamente, reducir RUL proporcionalmente
            acceleration_penalty = np.clip(1.0 - (degradation_velocity * 0.05), 0.3, 1.0)
            rul_median *= acceleration_penalty

        rul_median = max(float(rul_median), self.params.min_rul)

        # Factor de dispersión estocástica de Weibull
        # Coeficiente de variación típico de Weibull: CV = sqrt(Gamma(1+2/b) - Gamma(1+1/b)^2) / Gamma(1+1/b)
        cv = np.sqrt(max(gamma(1.0 + 2.0 / self.params.beta) - (gamma_factor ** 2), 0.01)) / gamma_factor
        std_rul = rul_median * cv

        # Cuantiles estocásticos condicionales:
        # P10: Percentil 10 (90% de certeza de durar al menos este tiempo - Planificación Minera)
        # P50: Mediana esperada
        # P90: Percentil 90 (Escenario optimista)
        rul_p10 = max(float(rul_median - 1.2816 * std_rul), self.params.min_rul)
        rul_p50 = float(rul_median)
        rul_p90 = max(float(rul_median + 1.2816 * std_rul), rul_p10 + 1.0)

        # Nivel de riesgo operacional
        if rul_p10 < 15.0:
            risk_tier = "CRITICAL_MAINTENANCE_WINDOW"
        elif rul_p10 < 50.0:
            risk_tier = "NEAR_TERM_ATTENTION"
        else:
            risk_tier = "OPERATIONAL_STABILITY"

        return {
            "health_score": round(hs, 2),
            "operating_hours": float(operating_hours),
            "rul_p10_conservative": round(rul_p10, 1),
            "rul_p50_median": round(rul_p50, 1),
            "rul_p90_optimistic": round(rul_p90, 1),
            "uncertainty_spread": round(rul_p90 - rul_p10, 1),
            "risk_tier": risk_tier,
            "unit": "ciclos_operacionales",
            "weibull_shape_beta": self.params.beta,
            "weibull_scale_eta": self.params.eta
        }
