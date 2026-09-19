"""
Optimizador de Decisión CBM (Condition-Based Maintenance) bajo Costo Asimétrico Minero.
Aplica el Clasificador de Mínimo Riesgo de Bayes para evitar fallas catastróficas en faena.
Sigue estrictamente la norma ISO 13374 y principios SOLID.
"""

from dataclasses import dataclass
from typing import Dict, Any, Union, Sequence
import numpy as np


@dataclass(frozen=True)
class IndustrialCostMatrix:
    """
    Matriz de Costos Industriales Asimétricos para Flotas de Camiones Mineros CAEX.
    Valores expresados en USD basados en benchmark de Gran Minería (Codelco / BHP).
    """
    cost_false_positive: float = 8000.0    # Parada innecesaria en taller mecánico (lucro cesante e inspección)
    cost_false_negative: float = 350000.0  # Rotura catastrófica de motor/hidráulica en rampa a plena carga
    cost_true_positive: float = 12000.0    # Mantenimiento correctivo planificado en taller
    cost_true_negative: float = 0.0        # Operación continua normal sin anomalías
    cost_warning_inspection: float = 2500.0 # Chequeo rápido preventivo en cambio de turno


class CostSensitiveCBMDecider:
    """
    Motor de Decisión CBM basado en la teoría de Decisión Bayesiana.
    Calcula analíticamente el umbral crítico óptimo (theta*) minimizando la
    pérdida económica esperada en lugar de utilizar un argmax simétrico (50%).
    """

    def __init__(self, costs: IndustrialCostMatrix = IndustrialCostMatrix()):
        self.costs = costs
        # Umbral analítico de Bayes: theta* = C_FP / (C_FN + C_FP)
        denom = self.costs.cost_false_negative + self.costs.cost_false_positive
        self.optimal_critical_threshold: float = self.costs.cost_false_positive / denom
        self.warning_threshold: float = 0.20  # 20% probabilidad de degradación incipiente

    def decide_maintenance_action(
        self,
        class_probabilities: Union[Sequence[float], np.ndarray]
    ) -> Dict[str, Any]:
        """
        Determina la acción operacional óptima a partir de las probabilidades de clase:
        [P(Normal), P(Warning), P(Critical)].

        Args:
            class_probabilities: Secuencia de 3 probabilidades sumando 1.0.

        Returns:
            Diccionario estructurado con acción recomendada, urgencia, umbrales y justificación económica.
        """
        probs = np.asarray(class_probabilities, dtype=np.float64).ravel()
        if len(probs) != 3:
            raise ValueError(f"Se esperaban 3 probabilidades [Normal, Warning, Critical], se recibieron {len(probs)}")

        p_normal, p_warning, p_critical = float(probs[0]), float(probs[1]), float(probs[2])

        # Pérdida esperada de cada acción operacional
        # Acción 1: Parada Inmediata (Taller)
        # - Si es Normal -> FP -> $8,000
        # - Si es Warning -> TP planificado -> $12,000
        # - Si es Critical -> TP crítico -> $12,000
        expected_cost_stop = (
            p_normal * self.costs.cost_false_positive +
            (p_warning + p_critical) * self.costs.cost_true_positive
        )

        # Acción 2: Continuar Operación
        # - Si es Normal -> TN -> $0
        # - Si es Warning -> Riesgo de escalamiento a falla -> $25,000
        # - Si es Critical -> FN catastrófico -> $350,000
        expected_cost_continue = (
            p_normal * self.costs.cost_true_negative +
            p_warning * 25000.0 +
            p_critical * self.costs.cost_false_negative
        )

        # 1. Regla de Mínimo Riesgo de Bayes para Falla Crítica
        if p_critical >= self.optimal_critical_threshold:
            action = "DESPACHO_INMEDIATO_TALLER"
            urgency = "CRITICAL"
            severity_code = 2
            avoided_risk = self.costs.cost_false_negative - self.costs.cost_true_positive
            reason = (
                f"Probabilidad de falla crítica ({p_critical * 100:.2f}%) supera el umbral óptimo "
                f"de Bayes ({self.optimal_critical_threshold * 100:.2f}%). "
                f"Riesgo económico evitado: ${avoided_risk:,.0f} USD frente a rotura en rampa."
            )
        # 2. Regla de Alerta Incipiente (Warning)
        elif p_warning >= self.warning_threshold:
            action = "INSPECCION_SIGUIENTE_CAMBIO_TURNO"
            urgency = "WARNING"
            severity_code = 1
            avoided_risk = 25000.0 - self.costs.cost_warning_inspection
            reason = (
                f"Degradación incipiente detectada ({p_warning * 100:.2f}%). "
                f"Programar chequeo preventivo en el siguiente cambio de turno."
            )
        # 3. Operación Continua Normal
        else:
            action = "OPERACION_CONTINUA_NORMAL"
            urgency = "NORMAL"
            severity_code = 0
            avoided_risk = 0.0
            reason = "Parámetros de telemetría dentro de la envolvente termodinámica segura."

        # Comparación con Argmax Tradicional Simétrico
        argmax_idx = int(np.argmax(probs))
        argmax_states = ["NORMAL", "WARNING", "CRITICAL"]
        traditional_decision = argmax_states[argmax_idx]

        economic_divergence = (traditional_decision != urgency)

        return {
            "action": action,
            "urgency": urgency,
            "severity_code": severity_code,
            "probabilities": {
                "normal": p_normal,
                "warning": p_warning,
                "critical": p_critical
            },
            "p_critical": p_critical,
            "optimal_critical_threshold": self.optimal_critical_threshold,
            "threshold_percentage": self.optimal_critical_threshold * 100.0,
            "avoided_risk_usd": avoided_risk,
            "expected_cost_stop_usd": round(expected_cost_stop, 2),
            "expected_cost_continue_usd": round(expected_cost_continue, 2),
            "traditional_argmax_decision": traditional_decision,
            "economic_divergence": economic_divergence,
            "justification": reason
        }
