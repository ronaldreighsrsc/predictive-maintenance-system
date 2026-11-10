"""
Tests de los componentes de realismo del simulador de sensores:
    1. Ruido autocorrelacionado AR(1) (vs. ruido blanco i.i.d.).
    2. Reseteos por mantención preventiva (desgaste en diente de sierra).
    3. Fallas de medición del sensor (datos faltantes, valores pegados, picos espurios).
Además verifica que el pipeline de features tolera datos faltantes sin fuga temporal.
"""
import numpy as np
import pandas as pd

from src.preprocessing.data_synthesizer import SensorDataSynthesizer
from src.preprocessing.feature_engineer import SensorFeatureEngineer


def _lag1_autocorr(x: np.ndarray) -> float:
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


# ---------------------------------------------------------------
# Compatibilidad hacia atrás
# ---------------------------------------------------------------
def test_defaults_preserve_original_dataset(tmp_path):
    """Sin activar componentes nuevos, el dataset es idéntico al original (reproducibilidad)."""
    legacy = SensorDataSynthesizer(n_equipment=5, cycles_per_equipment=100)
    explicit = SensorDataSynthesizer(
        n_equipment=5, cycles_per_equipment=100,
        noise_ar_coef=0.0, maintenance_interval=None, sensor_fault_rate=0.0,
    )
    df_a = legacy.generate(output_path=str(tmp_path / "a.csv"))
    df_b = explicit.generate(output_path=str(tmp_path / "b.csv"))
    pd.testing.assert_frame_equal(df_a, df_b)
    assert df_a.isnull().sum().sum() == 0


# ---------------------------------------------------------------
# 1. Ruido AR(1)
# ---------------------------------------------------------------
def test_white_noise_has_no_autocorrelation():
    synth = SensorDataSynthesizer(noise_ar_coef=0.0)
    temp = synth._generate_normal_readings(5000)['engine_temp']
    assert abs(_lag1_autocorr(temp)) < 0.05


def test_ar1_noise_is_autocorrelated_with_same_marginal_std():
    phi = 0.7
    synth = SensorDataSynthesizer(noise_ar_coef=phi)
    readings = synth._generate_normal_readings(20000)
    temp = readings['engine_temp']
    spec = SensorDataSynthesizer.SENSOR_SPECS['engine_temp']

    assert abs(_lag1_autocorr(temp) - phi) < 0.05
    # La dispersión marginal se mantiene: solo cambia la "memoria" del ruido
    assert abs(temp.std() - spec['normal_std']) / spec['normal_std'] < 0.10
    assert abs(temp.mean() - spec['normal_mean']) < 0.5


# ---------------------------------------------------------------
# 2. Reseteos por mantención
# ---------------------------------------------------------------
def test_maintenance_resets_wear_in_sawtooth_pattern():
    n_cycles, interval = 300, 60
    synth = SensorDataSynthesizer(maintenance_interval=interval)
    flat = {s: np.full(n_cycles, spec['normal_mean'], dtype=float)
            for s, spec in SensorDataSynthesizer.SENSOR_SPECS.items()}

    worn, events = synth._apply_maintenance_resets(flat, until_cycle=n_cycles)

    assert len(events) >= 3
    temp = worn['engine_temp']
    for ev in events:
        # Justo antes de la mantención hay desgaste acumulado; justo después vuelve a la base
        assert temp[ev - 1] > temp[ev]
        assert temp[ev] == SensorDataSynthesizer.SENSOR_SPECS['engine_temp']['normal_mean']
    # El desgaste empeora la presión de aceite (baja) antes de cada mantención
    assert worn['oil_pressure'][events[0] - 1] < SensorDataSynthesizer.SENSOR_SPECS['oil_pressure']['normal_mean']


def test_maintenance_does_not_touch_degradation_window():
    n_cycles = 200
    synth = SensorDataSynthesizer(maintenance_interval=40)
    flat = {s: np.zeros(n_cycles) for s in SensorDataSynthesizer.SENSOR_SPECS}

    worn, events = synth._apply_maintenance_resets(flat, until_cycle=120)

    assert all(ev < 120 for ev in events)
    for sensor in SensorDataSynthesizer.SENSOR_SPECS:
        assert np.all(worn[sensor][120:] == 0)


# ---------------------------------------------------------------
# 3. Fallas de medición del sensor
# ---------------------------------------------------------------
def test_sensor_faults_inject_missing_stuck_and_spikes():
    n_cycles = 2000
    synth = SensorDataSynthesizer(sensor_fault_rate=0.02)
    clean = synth._generate_normal_readings(n_cycles)
    original = {s: v.copy() for s, v in clean.items()}

    faulty = synth._inject_sensor_faults(clean, n_cycles)

    temp = faulty['engine_temp']
    spec = SensorDataSynthesizer.SENSOR_SPECS['engine_temp']
    # Datos faltantes (dropouts de telemetría)
    assert np.isnan(temp).sum() > 0
    # Picos espurios: lecturas imposibles para el ruido normal (> 5 sigmas)
    z = np.abs(temp - spec['normal_mean']) / spec['normal_std']
    assert np.nansum(z > 5) > 0
    # Valores pegados: tramos consecutivos idénticos que no existían en la señal limpia
    valid = temp[~np.isnan(temp)]
    assert np.sum(np.diff(valid) == 0) >= 3
    # La señal física original no se modifica (la falla es de medición, no del proceso)
    assert np.array_equal(original['engine_temp'], clean['engine_temp'])


def test_sensor_faults_do_not_alter_labels_or_rul(tmp_path):
    clean = SensorDataSynthesizer(n_equipment=6, cycles_per_equipment=100).generate(
        output_path=str(tmp_path / "clean.csv"))
    faulty = SensorDataSynthesizer(n_equipment=6, cycles_per_equipment=100,
                                   sensor_fault_rate=0.02).generate(
        output_path=str(tmp_path / "faulty.csv"))

    pd.testing.assert_series_equal(clean['machine_status'], faulty['machine_status'])
    pd.testing.assert_series_equal(clean['rul'], faulty['rul'])
    assert faulty[list(SensorDataSynthesizer.SENSOR_SPECS)].isnull().sum().sum() > 0


# ---------------------------------------------------------------
# Robustez del pipeline ante datos faltantes
# ---------------------------------------------------------------
def test_feature_engineer_imputes_missing_causally_per_equipment():
    """Un NaN se imputa con la última lectura válida del MISMO equipo (sin mirar el futuro)."""
    sensors = list(SensorDataSynthesizer.SENSOR_SPECS)
    rows = []
    for eq, base in [(0, 10.0), (1, 500.0)]:
        for cycle in range(1, 5):
            row = {'equipment_id': eq, 'cycle': cycle, 'operating_hours': cycle * 8}
            row.update({s: base + cycle for s in sensors})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.loc[(df.equipment_id == 0) & (df.cycle == 3), 'engine_temp'] = np.nan
    df.loc[(df.equipment_id == 1) & (df.cycle == 1), 'engine_temp'] = np.nan

    out = SensorFeatureEngineer().engineer_features(df)

    eq0 = out[out.equipment_id == 0].set_index('cycle')['engine_temp']
    eq1 = out[out.equipment_id == 1].set_index('cycle')['engine_temp']
    assert eq0[3] == 12.0          # forward-fill causal (valor del ciclo 2), no 14.0 del futuro
    assert eq1[1] == 502.0         # NaN inicial: primera lectura válida del mismo equipo, no del equipo 0
    assert out[sensors].isnull().sum().sum() == 0
