import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class SensorDataSynthesizer:
    """
    Generador de datos sintéticos de sensores de maquinaria pesada (camiones mineros).
    Simula lecturas de sensores con degradación progresiva antes de una falla,
    imitando el comportamiento real de equipos industriales.

    Patrones simulados:
        - Operación normal: lecturas estables con ruido gaussiano.
        - Degradación: drift gradual en múltiples sensores antes de la falla.
        - Falla: cambios abruptos en los últimos ciclos de operación.
        - Mantenimiento: reset de sensores después de intervención.
    """

    def __init__(self, n_equipment: int = 50, cycles_per_equipment: int = 300,
                 failure_rate: float = 0.30, random_state: int = 42):
        self.n_equipment = n_equipment
        self.cycles_per_equipment = cycles_per_equipment
        self.failure_rate = failure_rate
        self.rng = np.random.RandomState(random_state)

    # ---------------------------------------------------------------
    # Parámetros base de cada sensor (rango normal de operación)
    # ---------------------------------------------------------------
    SENSOR_SPECS = {
        'engine_temp':       {'normal_mean': 85,  'normal_std': 3,   'unit': '°C'},
        'oil_pressure':      {'normal_mean': 45,  'normal_std': 2,   'unit': 'PSI'},
        'vibration_level':   {'normal_mean': 2.5, 'normal_std': 0.3, 'unit': 'mm/s'},
        'rpm':               {'normal_mean': 1800,'normal_std': 50,  'unit': 'RPM'},
        'fuel_consumption':  {'normal_mean': 35,  'normal_std': 2,   'unit': 'L/h'},
        'coolant_temp':      {'normal_mean': 78,  'normal_std': 2.5, 'unit': '°C'},
        'hydraulic_pressure':{'normal_mean': 3200,'normal_std': 100, 'unit': 'PSI'},
    }

    def _generate_normal_readings(self, n_cycles: int) -> dict:
        """Genera lecturas de operación normal con ruido gaussiano."""
        readings = {}
        for sensor, specs in self.SENSOR_SPECS.items():
            readings[sensor] = self.rng.normal(
                specs['normal_mean'], specs['normal_std'], size=n_cycles
            )
        return readings

    def _apply_degradation(self, readings: dict, n_cycles: int,
                           failure_point: int) -> dict:
        """
        Aplica degradación progresiva a los sensores antes del punto de falla.
        El deterioro es gradual (drift lineal + ruido creciente) para simular
        desgaste mecánico real.
        """
        degradation_start = max(0, failure_point - int(n_cycles * 0.4))
        degradation_length = failure_point - degradation_start

        for cycle in range(degradation_start, min(failure_point, n_cycles)):
            progress = (cycle - degradation_start) / max(degradation_length, 1)

            # Temperatura del motor: sube con degradación
            readings['engine_temp'][cycle] += progress * 25 + self.rng.normal(0, progress * 3)
            # Presión de aceite: baja con degradación
            readings['oil_pressure'][cycle] -= progress * 15 + self.rng.normal(0, progress * 2)
            # Vibración: aumenta significativamente
            readings['vibration_level'][cycle] += progress * 5 + self.rng.normal(0, progress * 1.5)
            # RPM: se vuelve inestable
            readings['rpm'][cycle] += self.rng.normal(0, progress * 150)
            # Consumo de combustible: aumenta con ineficiencia
            readings['fuel_consumption'][cycle] += progress * 10 + self.rng.normal(0, progress * 2)
            # Refrigerante: sube junto con el motor
            readings['coolant_temp'][cycle] += progress * 15 + self.rng.normal(0, progress * 2)
            # Presión hidráulica: fluctúa
            readings['hydraulic_pressure'][cycle] += self.rng.normal(0, progress * 300)

        return readings

    def _assign_labels(self, n_cycles: int, failure_point: int) -> np.ndarray:
        """
        Asigna etiquetas de estado del equipo:
            0 = Normal
            1 = Warning (últimos 30% antes de la falla)
            2 = Critical (últimos 10% antes de la falla)
        """
        labels = np.zeros(n_cycles, dtype=int)

        if failure_point < n_cycles:
            warning_start = max(0, failure_point - int(n_cycles * 0.30))
            critical_start = max(0, failure_point - int(n_cycles * 0.10))

            labels[warning_start:critical_start] = 1   # Warning
            labels[critical_start:failure_point] = 2    # Critical

        return labels

    def _calculate_rul(self, n_cycles: int, failure_point: int) -> np.ndarray:
        """Calcula el Remaining Useful Life (RUL) en cada ciclo."""
        rul = np.zeros(n_cycles)
        for i in range(n_cycles):
            rul[i] = max(0, failure_point - i)
        return rul

    # ---------------------------------------------------------------
    # Orquestador público
    # ---------------------------------------------------------------
    def generate(self, output_path: str = "./data/raw/sensor_readings.csv") -> pd.DataFrame:
        """
        Pipeline principal de generación de datos de sensores.
        Retorna un DataFrame con lecturas de sensores para múltiples equipos.
        """
        print("🏭 Generando datos sintéticos de sensores industriales...")

        all_rows = []
        n_with_failure = int(self.n_equipment * self.failure_rate)

        for eq_id in range(self.n_equipment):
            n_cycles = self.rng.randint(
                int(self.cycles_per_equipment * 0.7),
                int(self.cycles_per_equipment * 1.3)
            )

            # Generar lecturas normales
            readings = self._generate_normal_readings(n_cycles)

            # Determinar si este equipo tiene falla
            has_failure = eq_id < n_with_failure
            if has_failure:
                failure_point = self.rng.randint(
                    int(n_cycles * 0.5), n_cycles
                )
                readings = self._apply_degradation(readings, n_cycles, failure_point)
            else:
                failure_point = n_cycles  # No falla

            # Asignar labels y RUL
            labels = self._assign_labels(n_cycles, failure_point)
            rul = self._calculate_rul(n_cycles, failure_point)

            # Construir DataFrame para este equipo
            for cycle in range(n_cycles):
                row = {
                    'equipment_id': eq_id,
                    'cycle': cycle + 1,
                    'operating_hours': (cycle + 1) * 8,  # 8 horas por ciclo
                }
                for sensor in self.SENSOR_SPECS:
                    row[sensor] = round(readings[sensor][cycle], 2)
                row['machine_status'] = labels[cycle]
                row['rul'] = int(rul[cycle])
                all_rows.append(row)

        df = pd.DataFrame(all_rows)

        # Resumen
        n_normal = (df['machine_status'] == 0).sum()
        n_warning = (df['machine_status'] == 1).sum()
        n_critical = (df['machine_status'] == 2).sum()

        print(f"  🚛 {self.n_equipment} equipos simulados")
        print(f"  📊 {len(df):,} lecturas de sensores generadas")
        print(f"     Normal: {n_normal:,} | Warning: {n_warning:,} | Critical: {n_critical:,}")

        # Guardar
        df.to_csv(output_path, index=False)
        print(f"  💾 Dataset guardado en: {output_path}")

        return df
