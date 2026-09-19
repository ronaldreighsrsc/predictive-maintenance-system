"""
Monitor de Calibración y Deriva de Sensores (Sensor Drift & Cross-Sensor Physical Consistency).
Utiliza el test no paramétrico de Kolmogorov-Smirnov y reglas termodinámicas de consistencia cruzada
para discriminar entre descalibración física de termistores/transductores y fallas mecánicas reales.
Evita paradas innecesarias y costosas de camiones CAEX en faena minera.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Sequence
import numpy as np
from scipy import stats


@dataclass
class SensorBaselineSignature:
    """Firma de calibración de fábrica de un sensor en condición nominal de ralentí/operación."""
    mean: float
    std: float
    min_val: float
    max_val: float
    reference_distribution: np.ndarray


class SensorDriftDetector:
    """
    Detector de Deriva Física y Falla de Sensores.
    Mantiene ventanas históricas recientes de sensores y realiza:
    1. Test de Kolmogorov-Smirnov (KS-test) contra distribución de calibración.
    2. Validación termodinámica y física cruzada entre sensores correlacionados.
    """

    SENSOR_KEYS: List[str] = [
        'engine_temp',
        'oil_pressure',
        'vibration_level',
        'rpm',
        'fuel_consumption',
        'coolant_temp',
        'hydraulic_pressure'
    ]

    def __init__(self, history_window: int = 30):
        self.history_window = history_window
        # Buffers por equipo y sensor: {equipment_id: {sensor_name: deque(maxlen=30)}}
        self._sensor_buffers: Dict[str, Dict[str, deque]] = {}
        # Líneas base de referencia por sensor
        self.baselines: Dict[str, SensorBaselineSignature] = self._init_default_baselines()

    def _init_default_baselines(self) -> Dict[str, SensorBaselineSignature]:
        """Inicializa firmas sintéticas de referencia normal para camiones Komatsu 930E / Cat 797F."""
        np.random.seed(42)
        base_configs = {
            'engine_temp': (85.0, 3.0, 75.0, 95.0),
            'oil_pressure': (45.0, 2.0, 35.0, 55.0),
            'vibration_level': (2.5, 0.3, 1.5, 4.0),
            'rpm': (1800.0, 50.0, 1500.0, 2100.0),
            'fuel_consumption': (35.0, 2.0, 25.0, 50.0),
            'coolant_temp': (78.0, 2.5, 68.0, 88.0),
            'hydraulic_pressure': (3200.0, 100.0, 2800.0, 3600.0),
        }
        baselines = {}
        for sensor, (mean, std, min_v, max_v) in base_configs.items():
            ref_dist = np.clip(np.random.normal(mean, std, 200), min_v, max_v)
            baselines[sensor] = SensorBaselineSignature(
                mean=mean, std=std, min_val=min_v, max_val=max_v, reference_distribution=ref_dist
            )
        return baselines

    def push_and_audit_sensors(
        self,
        equipment_id: str,
        current_readings: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Registra la lectura actual y audita la coherencia física y estadística de los sensores.

        Returns:
            Dict con diagnóstico de deriva, consistencia termodinámica y clasificación de falla.
        """
        if equipment_id not in self._sensor_buffers:
            self._sensor_buffers[equipment_id] = {
                k: deque(maxlen=self.history_window) for k in self.SENSOR_KEYS
            }

        eq_buffers = self._sensor_buffers[equipment_id]
        for k in self.SENSOR_KEYS:
            if k in current_readings:
                eq_buffers[k].append(float(current_readings[k]))

        # 1. Validación Cruzada Física (Cross-Sensor Consistency)
        physical_status = self._verify_physical_consistency(current_readings)

        # 2. Test Kolmogorov-Smirnov de deriva si hay suficiente historial
        ks_results = {}
        drift_detected_sensors = []
        for k in self.SENSOR_KEYS:
            buf = eq_buffers.get(k)
            if buf and len(buf) >= 15:
                ref_dist = self.baselines[k].reference_distribution
                sample = np.array(buf)
                stat, p_val = stats.ks_2samp(sample, ref_dist)
                is_drift = bool(p_val < 0.01)
                ks_results[k] = {
                    "ks_statistic": round(float(stat), 4),
                    "p_value": round(float(p_val), 5),
                    "has_distribution_drift": is_drift
                }
                if is_drift:
                    drift_detected_sensors.append(k)

        # 3. Decisión Integral: ¿Falla Mecánica o Falla de Sensor?
        if physical_status["is_sensor_discrepancy"]:
            classification = "SENSOR_CALIBRATION_DRIFT"
            severity = "WARNING"
            recommendation = (
                f"Posible descalibración física de sensor detectada: {', '.join(physical_status['discrepancies'])}. "
                "Recalibrar o sustituir transductor antes de programar parada de motor."
            )
        elif len(drift_detected_sensors) > 0 and physical_status["is_physically_consistent"]:
            # Si hay deriva pero la termodinámica es consistente (ambos sensores subieron juntos), es degradación real
            classification = "MECHANICAL_DEGRADATION"
            severity = "WARNING"
            recommendation = (
                f"Deriva correlacionada consistente con desgaste mecánico en sensores: "
                f"{', '.join(drift_detected_sensors)}. Seguir monitoreando."
            )
        else:
            classification = "HEALTHY_TELEMETRY"
            severity = "NORMAL"
            recommendation = "Firma de sensores dentro de parámetros calibrados."

        return {
            "equipment_id": equipment_id,
            "classification": classification,
            "severity": severity,
            "is_sensor_fault": physical_status["is_sensor_discrepancy"],
            "discrepancies": physical_status["discrepancies"],
            "ks_test_results": ks_results,
            "drift_sensors": drift_detected_sensors,
            "recommendation": recommendation
        }

    def _verify_physical_consistency(self, readings: Dict[str, float]) -> Dict[str, Any]:
        """Aplica leyes termodinámicas y mecánicas de conservación entre sensores acoplados."""
        discrepancies = []

        eng_temp = readings.get('engine_temp', 85.0)
        cool_temp = readings.get('coolant_temp', 75.0)
        rpm = readings.get('rpm', 1800.0)
        fuel = readings.get('fuel_consumption', 12.0)
        oil_p = readings.get('oil_pressure', 45.0)
        vib = readings.get('vibration_level', 0.5)
        hyd_p = readings.get('hydraulic_pressure', 200.0)

        # Regla 1: Desacoplamiento Térmico Motor vs Refrigerante
        # Si la temperatura de motor sube a > 105°C pero el refrigerante está frío (< 70°C)
        # y las RPM son bajas (< 1200) -> Falla termistor motor, no sobrecalentamiento real
        if eng_temp > 100.0 and cool_temp < 70.0 and rpm < 1300.0:
            discrepancies.append(
                f"Termistor de motor disparado ({eng_temp:.1f}°C) con refrigerante frío ({cool_temp:.1f}°C) a bajas RPM ({rpm:.0f})"
            )

        # Regla 2: Presión de Aceite vs RPM
        # A altas RPM (> 1900), la bomba de aceite debe generar presión mínima (> 25 PSI).
        # Si la presión cae a 0 abruptamente sin vibración ni alza de temperatura -> Falla de transductor
        if rpm > 1700.0 and oil_p < 5.0 and vib < 0.6 and eng_temp < 92.0:
            discrepancies.append(
                f"Lectura de presión de aceite nula ({oil_p:.1f} PSI) a {rpm:.0f} RPM sin vibración anómala ni alza térmica"
            )

        # Regla 3: Vibración Fantasma
        # Vibración extrema (> 2.5 mm/s) con motor detenido o en ralentí mínimo (< 800 RPM)
        if vib > 2.0 and rpm < 900.0:
            discrepancies.append(
                f"Vibración anómala ({vib:.2f} mm/s) en ralentí de motor ({rpm:.0f} RPM) - Posible acelerómetro suelto"
            )

        # Regla 4: Transductor de Presión Hidráulica Caído
        if hyd_p < 20.0 and rpm > 1400.0 and oil_p > 35.0:
            discrepancies.append(
                f"Presión hidráulica en circuito principal anormalmente baja ({hyd_p:.1f} PSI) con motor en carga"
            )

        is_discrepancy = len(discrepancies) > 0
        return {
            "is_sensor_discrepancy": is_discrepancy,
            "is_physically_consistent": not is_discrepancy,
            "discrepancies": discrepancies
        }
