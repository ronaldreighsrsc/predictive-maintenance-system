import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class SensorDataSynthesizer:
    """
    Generador de datos sintéticos de sensores de maquinaria pesada (camiones mineros).
    Simula lecturas de sensores con degradación progresiva antes de una falla,
    imitando el comportamiento real de equipos industriales.

    Patrones simulados (siempre activos):
        - Operación normal: lecturas estacionarias con ruido gaussiano.
        - Degradación: drift lineal + ruido creciente en múltiples sensores antes de la falla.

    Componentes de realismo (opcionales, desactivados por defecto para preservar
    la reproducibilidad del dataset publicado):
        - noise_ar_coef: ruido autocorrelacionado AR(1). El ruido real tiene "memoria"
          (si un ciclo vibra mucho, el siguiente probablemente también).
        - maintenance_interval: mantención preventiva periódica. Un desgaste leve se
          acumula y vuelve a la línea base tras cada intervención (diente de sierra).
        - sensor_fault_rate: fallas de medición (datos faltantes, valores pegados y
          picos espurios). Afectan la lectura, no el estado físico ni las etiquetas.

    Cada componente opcional usa su propio generador aleatorio: activarlo no altera
    el resto de la flota, lo que permite estudios de ablación con/sin el componente.
    """

    # Firma de desgaste leve entre mantenciones (dirección × desviaciones estándar al
    # final del intervalo). Se mantiene sub-umbral: el equipo sigue etiquetado como Normal.
    WEAR_SIGNATURE = {
        'engine_temp': +1.0,
        'oil_pressure': -1.0,
        'vibration_level': +1.0,
        'fuel_consumption': +1.0,
        'coolant_temp': +1.0,
    }

    def __init__(self, n_equipment: int = 50, cycles_per_equipment: int = 300,
                 failure_rate: float = 0.30, random_state: int = 42,
                 noise_ar_coef: float = 0.0,
                 maintenance_interval: int | None = None,
                 sensor_fault_rate: float = 0.0):
        if not 0.0 <= noise_ar_coef < 1.0:
            raise ValueError("noise_ar_coef debe estar en [0, 1) para un AR(1) estacionario.")
        if not 0.0 <= sensor_fault_rate < 1.0:
            raise ValueError("sensor_fault_rate debe estar en [0, 1).")
        if maintenance_interval is not None and maintenance_interval < 2:
            raise ValueError("maintenance_interval debe ser >= 2 ciclos.")

        self.n_equipment = n_equipment
        self.cycles_per_equipment = cycles_per_equipment
        self.failure_rate = failure_rate
        self.noise_ar_coef = noise_ar_coef
        self.maintenance_interval = maintenance_interval
        self.sensor_fault_rate = sensor_fault_rate
        self.rng = np.random.RandomState(random_state)
        self._maintenance_rng = np.random.RandomState(random_state + 1)
        self._fault_rng = np.random.RandomState(random_state + 2)

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
        """
        Genera lecturas de operación normal con ruido gaussiano.

        Con noise_ar_coef = 0 el ruido es blanco (i.i.d.). Con noise_ar_coef = φ > 0
        sigue un proceso AR(1):  e_t = φ·e_{t-1} + σ·√(1-φ²)·z_t,  z_t ~ N(0, 1).
        El factor √(1-φ²) mantiene la desviación estándar marginal igual a σ:
        solo cambia la memoria temporal del ruido, no su magnitud.
        """
        readings = {}
        phi = self.noise_ar_coef
        for sensor, specs in self.SENSOR_SPECS.items():
            if phi == 0.0:
                readings[sensor] = self.rng.normal(
                    specs['normal_mean'], specs['normal_std'], size=n_cycles
                )
                continue

            z = self.rng.normal(0.0, 1.0, size=n_cycles)
            innovation_std = specs['normal_std'] * np.sqrt(1.0 - phi ** 2)
            noise = np.empty(n_cycles)
            noise[0] = specs['normal_std'] * z[0]  # arranque en la distribución estacionaria
            for t in range(1, n_cycles):
                noise[t] = phi * noise[t - 1] + innovation_std * z[t]
            readings[sensor] = specs['normal_mean'] + noise
        return readings

    @staticmethod
    def _degradation_start(n_cycles: int, failure_point: int) -> int:
        """Ciclo en que comienza la degradación (40% de la vida antes de la falla)."""
        return max(0, failure_point - int(n_cycles * 0.4))

    def _apply_degradation(self, readings: dict, n_cycles: int,
                           failure_point: int) -> dict:
        """
        Aplica degradación progresiva a los sensores antes del punto de falla.
        El deterioro es gradual (drift lineal + ruido creciente) para simular
        desgaste mecánico real.
        """
        degradation_start = self._degradation_start(n_cycles, failure_point)
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

    def _apply_maintenance_resets(self, readings: dict,
                                  until_cycle: int) -> tuple[dict, list[int]]:
        """
        Superpone desgaste leve en diente de sierra con mantenciones preventivas.

        Entre mantenciones el desgaste crece linealmente hasta WEAR_SIGNATURE·σ y,
        en cada ciclo de mantención, vuelve a cero (caída abrupta a la línea base).
        Los intervalos tienen jitter de ±20% (la planificación real no es exacta).
        Solo actúa antes de `until_cycle` para no interferir con la ventana de
        degradación hacia la falla.

        Retorna (lecturas_con_desgaste, ciclos_de_mantención).
        """
        worn = {sensor: values.copy() for sensor, values in readings.items()}
        events: list[int] = []
        base_interval = self.maintenance_interval

        last_event = 0
        while True:
            interval = self._maintenance_rng.randint(
                int(base_interval * 0.8), int(base_interval * 1.2) + 1
            )
            next_event = last_event + interval
            segment_end = min(next_event, until_cycle)
            progress = (np.arange(last_event, segment_end) - last_event) / interval

            for sensor, direction in self.WEAR_SIGNATURE.items():
                std = self.SENSOR_SPECS[sensor]['normal_std']
                worn[sensor][last_event:segment_end] += direction * std * progress

            if next_event >= until_cycle:
                break
            events.append(next_event)
            last_event = next_event

        return worn, events

    def _inject_sensor_faults(self, readings: dict, n_cycles: int) -> dict:
        """
        Inyecta fallas de MEDICIÓN (el estado físico del equipo no cambia):
            - Valores pegados: el sensor repite su última lectura por 5-15 ciclos.
            - Picos espurios: interferencia eléctrica de 6-10σ en un solo ciclo.
            - Datos faltantes: pérdida de paquetes de telemetría (NaN).

        `sensor_fault_rate` es la probabilidad de dato faltante por lectura; los
        picos ocurren a la mitad de esa tasa y los tramos pegados a un quinto.
        Retorna un diccionario nuevo; no muta las lecturas de entrada.
        """
        rate = self.sensor_fault_rate
        rng = self._fault_rng
        faulty = {}

        for sensor, values in readings.items():
            signal = values.astype(float).copy()
            std = self.SENSOR_SPECS[sensor]['normal_std']

            # 1) Valores pegados (stuck-at)
            for start in np.flatnonzero(rng.rand(n_cycles) < rate / 5):
                length = rng.randint(5, 16)
                signal[start:start + length] = signal[start]

            # 2) Picos espurios (EMI)
            spikes = rng.rand(n_cycles) < rate / 2
            signs = np.where(rng.rand(n_cycles) < 0.5, -1.0, 1.0)
            signal[spikes] += (signs * rng.uniform(6, 10, n_cycles) * std)[spikes]

            # 3) Datos faltantes (dropouts)
            signal[rng.rand(n_cycles) < rate] = np.nan

            faulty[sensor] = signal

        return faulty

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

            # Mantención preventiva: solo antes de la ventana de degradación
            if self.maintenance_interval is not None:
                until = (self._degradation_start(n_cycles, failure_point)
                         if has_failure else n_cycles)
                readings, _ = self._apply_maintenance_resets(readings, until)

            # Fallas de medición: última capa (afecta lecturas, no etiquetas)
            if self.sensor_fault_rate > 0.0:
                readings = self._inject_sensor_faults(readings, n_cycles)

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
