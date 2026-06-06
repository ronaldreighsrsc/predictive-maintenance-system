import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class MaintenanceIsolationForest:
    """
    Modelo baseline de Isolation Forest para detección de anomalías
    en datos de sensores industriales.

    Método no supervisado que aísla lecturas de sensores anómalas
    sin necesidad de labels. Sirve como benchmark contra los modelos
    más complejos (Autoencoder y XGBoost) del sistema.
    """

    def __init__(self, contamination: float = 0.10, n_estimators: int = 200,
                 random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X_train: np.ndarray) -> None:
        """Entrena el Isolation Forest con los datos de sensores."""
        print("  🌲 Entrenando Isolation Forest (baseline no supervisado)...")

        X_scaled = self.scaler.fit_transform(X_train)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)

        print(f"  ✅ Isolation Forest entrenado ({self.n_estimators} árboles, "
              f"contamination={self.contamination:.2%})")

    def predict(self, X: np.ndarray) -> tuple:
        """
        Predice anomalías en lecturas de sensores.
        Convierte: -1 (anomalía) → 1, 1 (normal) → 0
        Retorna: (predicciones binarias, anomaly scores)
        """
        X_scaled = self.scaler.transform(X)
        raw_predictions = self.model.predict(X_scaled)
        predictions = np.where(raw_predictions == -1, 1, 0)
        anomaly_scores = -self.model.decision_function(X_scaled)
        return predictions, anomaly_scores
