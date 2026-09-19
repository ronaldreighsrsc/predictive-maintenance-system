"""
Buffer de Ventana Deslizante en Memoria (O(1)) para Telemetría de Camiones Mineros CAEX.
Convierte 7 lecturas crudas del bus CAN en el vector completo de 68 features
utilizado por los modelos de Machine Learning y Deep Learning en producción.
Cumple con principios SOLID y arquitectura para streaming concurrente de baja latencia.
"""

from collections import deque
import threading
from typing import Dict, List, Optional, Union, Tuple
import numpy as np


class EquipmentSlidingBuffer:
    """
    Gestor de búfer circular en memoria RAM para flotas mineras.
    Mantiene un historial deslizante de longitud fija (por defecto 20 ciclos)
    por cada equipo (`equipment_id`), extrayendo características estadísticas,
    deltas temporales, promedios exponenciales (EMA) y relaciones físicas cross-sensor.
    """

    SENSOR_NAMES: List[str] = [
        'engine_temp',
        'oil_pressure',
        'vibration_level',
        'rpm',
        'fuel_consumption',
        'coolant_temp',
        'hydraulic_pressure'
    ]

    WINDOWS: List[int] = [5, 10, 20]

    def __init__(self, window_size: int = 20):
        """
        Inicializa el gestor de búferes.

        Args:
            window_size: Tamaño máximo de ventana histórica por equipo (mínimo 20).
        """
        self.window_size = max(window_size, 20)
        # Diccionario en RAM: {equipment_id: deque(maxlen=window_size)}
        # Cada elemento del deque almacena una tupla: (sensor_array_7, operating_hours)
        self.buffers: Dict[str, deque] = {}
        self.total_ingested: Dict[str, int] = {}
        self._lock = threading.Lock()

        # Pesos precalculados para EMA con span=10 (alpha = 2 / (span + 1) = 2/11)
        span = 10
        weights = [(1.0 - 2.0 / (span + 1)) ** i for i in reversed(range(10))]
        self._ema_weights = np.array(weights, dtype=np.float32)
        self._ema_weights /= self._ema_weights.sum()

    def push_reading(
        self,
        equipment_id: str,
        raw_sensors: Union[List[float], np.ndarray, Dict[str, float]],
        operating_hours: float = 0.0
    ) -> np.ndarray:
        """
        Ingesta una lectura cruda del bus CAN y genera el vector de 68 features en O(1).

        Args:
            equipment_id: Identificador único del camión (ej. "CAEX-104").
            raw_sensors: 7 lecturas de sensores base en orden canónico o diccionario.
            operating_hours: Horas acumuladas de operación del motor.

        Returns:
            np.ndarray de dimensión (1, 68) con dtype float32.
        """
        sensor_values = self._normalize_sensors(raw_sensors)

        with self._lock:
            if equipment_id not in self.buffers:
                self.buffers[equipment_id] = deque(maxlen=self.window_size)
                self.total_ingested[equipment_id] = 0

            buf = self.buffers[equipment_id]
            buf.append((sensor_values, float(operating_hours)))
            self.total_ingested[equipment_id] += 1

            # Snapshot del historial para cálculo vectorial
            history_sensors = [item[0] for item in buf]
            last_operating_hours = buf[-1][1]

        # Convertir a matriz numpy: shape (L, 7)
        history_arr = np.array(history_sensors, dtype=np.float32)
        history_len = len(history_arr)

        # Si aún no se acumulan suficientes ciclos, rellenar el inicio con la primera lectura
        if history_len < self.window_size:
            pad_rows = self.window_size - history_len
            padding = np.repeat(history_arr[:1], pad_rows, axis=0)
            history_arr = np.vstack([padding, history_arr])

        # history_arr ahora tiene forma garantizada (window_size, 7)
        current = history_arr[-1]
        previous = history_arr[-2]

        feature_list: List[float] = []

        # 1. Sensores Base (7 features)
        feature_list.extend(current.tolist())

        # 2. Operating Hours (1 feature)
        feature_list.append(float(last_operating_hours))

        # 3. Rolling Statistics (42 features: 7 sensores x 3 ventanas x 2 estadísticas)
        # Sigue exactamente el orden de SensorFeatureEngineer:
        # for sensor in sensor_cols: for w in [5, 10, 20]: mean, std
        for s_idx in range(7):
            sensor_series = history_arr[:, s_idx]
            for w in self.WINDOWS:
                sub_window = sensor_series[-w:]
                feature_list.append(float(np.mean(sub_window)))
                feature_list.append(float(np.std(sub_window) + 1e-6))

        # 4. Deltas instantáneos (7 features: tasa de cambio respecto al ciclo anterior)
        deltas = current - previous
        feature_list.extend(deltas.tolist())

        # 5. Exponential Moving Averages (7 features: EMA-10)
        recent_10 = history_arr[-10:]
        ema_values = np.dot(self._ema_weights, recent_10)
        feature_list.extend(ema_values.tolist())

        # 6. Ratios de interacción física y termodinámica (4 features)
        # engine_temp (0), oil_pressure (1), vibration_level (2), rpm (3),
        # fuel_consumption (4), coolant_temp (5), hydraulic_pressure (6)
        temp_oil_ratio = current[0] / (current[1] if abs(current[1]) > 1e-4 else 1.0)
        temp_coolant_diff = current[0] - current[5]
        rpm_norm = (current[3] / 1000.0) if abs(current[3]) > 1e-4 else 1.0
        vibration_per_rpm = current[2] / rpm_norm
        fuel_efficiency = current[4] / rpm_norm

        feature_list.extend([
            float(temp_oil_ratio),
            float(temp_coolant_diff),
            float(vibration_per_rpm),
            float(fuel_efficiency)
        ])

        feature_vector = np.array(feature_list, dtype=np.float32).reshape(1, -1)
        return feature_vector

    def _normalize_sensors(
        self,
        raw_sensors: Union[List[float], np.ndarray, Dict[str, float]]
    ) -> List[float]:
        """Convierte la entrada a lista de 7 valores flotantes en orden estándar."""
        if isinstance(raw_sensors, dict):
            try:
                return [float(raw_sensors[k]) for k in self.SENSOR_NAMES]
            except KeyError as e:
                raise ValueError(f"Falta el sensor requerido en la lectura: {e}")

        values = list(raw_sensors)
        if len(values) != 7:
            raise ValueError(
                f"Se esperaban 7 sensores crudos ({self.SENSOR_NAMES}), pero se recibieron {len(values)}"
            )
        return [float(v) for v in values]

    def get_buffer_status(self, equipment_id: str) -> Dict[str, Union[str, int, bool]]:
        """Retorna el estado de calentamiento y tamaño de búfer de un equipo."""
        with self._lock:
            buf = self.buffers.get(equipment_id)
            total = self.total_ingested.get(equipment_id, 0)
            cur_len = len(buf) if buf else 0

        return {
            "equipment_id": equipment_id,
            "buffer_length": cur_len,
            "capacity": self.window_size,
            "is_warmed_up": cur_len >= self.window_size,
            "total_ingested_cycles": total
        }

    def reset_equipment(self, equipment_id: str) -> bool:
        """Reinicia el historial de un equipo (ej. tras mantención de taller)."""
        with self._lock:
            if equipment_id in self.buffers:
                del self.buffers[equipment_id]
                self.total_ingested[equipment_id] = 0
                return True
            return False

    def get_active_equipments(self) -> List[str]:
        """Retorna la lista de identificadores de equipos activos en memoria."""
        with self._lock:
            return list(self.buffers.keys())
