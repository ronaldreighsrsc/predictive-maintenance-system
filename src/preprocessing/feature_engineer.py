import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class SensorFeatureEngineer:
    """
    Ingeniería de features para mantenimiento predictivo.
    Calcula variables derivadas de series temporales de sensores
    que capturan tendencias de degradación y patrones pre-falla.

    Features generadas por sensor:
        - Rolling mean (ventanas de 5, 10, 20 ciclos)
        - Rolling std (volatilidad del sensor)
        - Trend (pendiente lineal en ventana móvil)
        - Delta (cambio respecto al ciclo anterior)
        - Exponential Moving Average (EMA)
    """

    def __init__(self, windows: list = None):
        self.windows = windows or [5, 10, 20]
        self.scaler = StandardScaler()
        self.sensor_cols = [
            'engine_temp', 'oil_pressure', 'vibration_level',
            'rpm', 'fuel_consumption', 'coolant_temp', 'hydraulic_pressure'
        ]

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline principal de feature engineering por equipo.
        Aplica transformaciones de series temporales a cada sensor.
        """
        print("🔧 Iniciando Feature Engineering de sensores...")
        df = df.copy()
        df = df.sort_values(['equipment_id', 'cycle']).reset_index(drop=True)

        # Paso 1: Rolling statistics
        df = self._add_rolling_stats(df)
        print("  ✅ Rolling statistics calculadas")

        # Paso 2: Delta features (rate of change)
        df = self._add_delta_features(df)
        print("  ✅ Delta features calculadas")

        # Paso 3: Exponential Moving Average
        df = self._add_ema_features(df)
        print("  ✅ Exponential Moving Average calculada")

        # Paso 4: Cross-sensor features
        df = self._add_cross_sensor_features(df)
        print("  ✅ Features cross-sensor creadas")

        # Limpiar NaN generados por rolling
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

        print(f"  📐 Dimensiones finales: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        return df

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula media y desviación estándar móvil por equipo y sensor."""
        for sensor in self.sensor_cols:
            for w in self.windows:
                df[f'{sensor}_roll_mean_{w}'] = df.groupby('equipment_id')[sensor].transform(
                    lambda x: x.rolling(window=w, min_periods=1).mean()
                )
                df[f'{sensor}_roll_std_{w}'] = df.groupby('equipment_id')[sensor].transform(
                    lambda x: x.rolling(window=w, min_periods=1).std()
                )
        return df

    def _add_delta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula la tasa de cambio respecto al ciclo anterior."""
        for sensor in self.sensor_cols:
            df[f'{sensor}_delta'] = df.groupby('equipment_id')[sensor].transform(
                lambda x: x.diff()
            )
            df[f'{sensor}_delta_pct'] = df.groupby('equipment_id')[sensor].transform(
                lambda x: x.pct_change()
            )
        return df

    def _add_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula Exponential Moving Average (captura tendencia reciente)."""
        for sensor in self.sensor_cols:
            df[f'{sensor}_ema_10'] = df.groupby('equipment_id')[sensor].transform(
                lambda x: x.ewm(span=10, adjust=False).mean()
            )
        return df

    def _add_cross_sensor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de interacción entre sensores que capturan correlaciones de falla."""
        # Ratio temperatura motor / presión de aceite (indicador de desgaste)
        df['temp_oil_ratio'] = df['engine_temp'] / df['oil_pressure'].replace(0, 1)

        # Diferencia entre temperatura motor y refrigerante (eficiencia de enfriamiento)
        df['temp_coolant_diff'] = df['engine_temp'] - df['coolant_temp']

        # Vibración normalizada por RPM (vibración excesiva para las RPM dadas)
        df['vibration_per_rpm'] = df['vibration_level'] / (df['rpm'].replace(0, 1) / 1000)

        # Eficiencia de combustible (consumo / RPM)
        df['fuel_efficiency'] = df['fuel_consumption'] / (df['rpm'].replace(0, 1) / 1000)

        return df

    def get_feature_columns(self) -> list:
        """Retorna la lista de features seleccionadas para el modelado."""
        features = list(self.sensor_cols)
        features.append('operating_hours')

        # Rolling stats
        for sensor in self.sensor_cols:
            for w in self.windows:
                features.append(f'{sensor}_roll_mean_{w}')
                features.append(f'{sensor}_roll_std_{w}')

        # Delta
        for sensor in self.sensor_cols:
            features.append(f'{sensor}_delta')

        # EMA
        for sensor in self.sensor_cols:
            features.append(f'{sensor}_ema_10')

        # Cross-sensor
        features.extend([
            'temp_oil_ratio', 'temp_coolant_diff',
            'vibration_per_rpm', 'fuel_efficiency'
        ])

        return features

    def scale_features(self, df: pd.DataFrame, feature_cols: list,
                       fit: bool = True) -> np.ndarray:
        """Escala las features con StandardScaler, limpiando Infs y NaNs."""
        X = df[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
