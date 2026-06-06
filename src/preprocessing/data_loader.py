import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")


class SensorDataLoader:
    """
    Módulo de carga y validación de datos de sensores industriales.
    Soporta tanto datos sintéticos generados por SensorDataSynthesizer
    como datasets externos (e.g., NASA C-MAPSS Turbofan Engine Degradation).
    """

    def __init__(self, raw_data_path: str = "./data/raw/"):
        self.raw_data_path = raw_data_path

    def load_sensor_data(self, filename: str = "sensor_readings.csv") -> pd.DataFrame:
        """
        Carga el dataset de sensores desde un archivo CSV.
        Aplica validaciones de integridad y reporta estadísticas.
        """
        filepath = os.path.join(self.raw_data_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"❌ No se encontró el archivo: {filepath}\n"
                f"   Ejecuta primero 'main_preprocessing.py' para generar los datos sintéticos."
            )

        print(f"📂 Cargando datos de sensores desde: {filepath}")
        df = pd.read_csv(filepath)

        # Validación de columnas requeridas
        required_cols = [
            'equipment_id', 'cycle', 'operating_hours',
            'engine_temp', 'oil_pressure', 'vibration_level',
            'rpm', 'fuel_consumption', 'coolant_temp',
            'hydraulic_pressure', 'machine_status', 'rul'
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"❌ Columnas faltantes en el dataset: {missing}")

        self._print_summary(df)
        return df

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Imprime un resumen ejecutivo del dataset cargado."""
        n_equipment = df['equipment_id'].nunique()
        n_readings = len(df)
        n_normal = (df['machine_status'] == 0).sum()
        n_warning = (df['machine_status'] == 1).sum()
        n_critical = (df['machine_status'] == 2).sum()

        print(f"  📊 Resumen del dataset de sensores:")
        print(f"     Total de lecturas: {n_readings:,}")
        print(f"     Equipos únicos: {n_equipment}")
        print(f"     Normal: {n_normal:,} ({n_normal/n_readings*100:.1f}%)")
        print(f"     Warning: {n_warning:,} ({n_warning/n_readings*100:.1f}%)")
        print(f"     Critical: {n_critical:,} ({n_critical/n_readings*100:.1f}%)")
        print(f"     RUL promedio: {df['rul'].mean():.0f} ciclos")

        # Check de NaN
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            print(f"  ⚠️  Valores nulos detectados: {nan_count}")
        else:
            print(f"     ✅ Sin valores nulos")
