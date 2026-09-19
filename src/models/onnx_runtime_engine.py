"""
Motor de Inferencia Optimizado con ONNX Runtime y Fallback Transparente.
Permite inferencias de ultra baja latencia (< 2 ms) para entornos industriales de borde (Edge Gateway)
y minimiza la huella de memoria RAM en contenedores Docker de producción.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

logger = logging.getLogger("predictive_maintenance.onnx")


class ONNXRuntimeEngine:
    """
    Gestiona sesiones de ONNX Runtime para modelos tabulares y redes neuronales,
    proporcionando desacoplamiento y fallback automático a los modelos nativos
    de scikit-learn/Keras si el runtime ONNX o el artefacto .onnx no estuvieran presentes.
    """

    def __init__(self, models_dir: str = "./models/saved_models"):
        self.models_dir = models_dir
        self.onnx_available = False
        self.sessions: Dict[str, Any] = {}

        try:
            import onnxruntime as ort
            # Configuración para mínima latencia en CPU
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._ort = ort
            self._session_opts = opts
            self.onnx_available = True
        except ImportError:
            logger.warning("onnxruntime no está instalado. Operando en modo nativo (fallback).")

        self._load_available_onnx_models()

    def _load_available_onnx_models(self) -> None:
        """Carga en memoria los modelos .onnx presentes en el directorio."""
        if not self.onnx_available or not os.path.exists(self.models_dir):
            return

        for filename in os.listdir(self.models_dir):
            if filename.endswith(".onnx"):
                model_name = filename[:-5]
                filepath = os.path.join(self.models_dir, filename)
                try:
                    session = self._ort.InferenceSession(filepath, self._session_opts, providers=['CPUExecutionProvider'])
                    self.sessions[model_name] = session
                    logger.info(f"Modelo ONNX cargado exitosamente: {model_name} ({filepath})")
                except Exception as e:
                    logger.warning(f"No se pudo cargar {filename} en ONNX Runtime: {e}")

    def has_model(self, model_name: str) -> bool:
        """Indica si el modelo ONNX está activo en memoria."""
        return model_name in self.sessions

    def predict_xgboost(
        self,
        features_68: np.ndarray,
        fallback_model: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ejecuta inferencia multi-clase de XGBoost.
        Retorna (predicción_clase, probabilidades).
        """
        if self.has_model("xgb"):
            session = self.sessions["xgb"]
            input_name = session.get_inputs()[0].name
            # ONNX expects float32
            X_input = features_68.astype(np.float32)
            outputs = session.run(None, {input_name: X_input})
            # Salida típica de clasificador ONNX: [labels, probabilities_dicts_or_tensor]
            if len(outputs) >= 2:
                labels = outputs[0]
                probs = outputs[1]
                if isinstance(probs, list) and len(probs) > 0 and isinstance(probs[0], dict):
                    # formato lista de diccionarios {0: p0, 1: p1, 2: p2}
                    prob_mat = np.array([[d[i] for i in sorted(d.keys())] for d in probs], dtype=np.float32)
                else:
                    prob_mat = np.asarray(probs, dtype=np.float32)
                return np.asarray(labels).astype(int), prob_mat
            else:
                probs = np.asarray(outputs[0], dtype=np.float32)
                preds = np.argmax(probs, axis=1)
                return preds, probs

        if fallback_model is not None:
            return fallback_model.predict(features_68)

        raise RuntimeError("No se encontró sesión ONNX 'xgb' ni modelo de fallback nativo.")

    def predict_autoencoder_health(
        self,
        features_68: np.ndarray,
        fallback_model: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ejecuta el Deep Autoencoder para calcular el Health Score (0 a 100).
        Retorna (is_anomaly, health_scores).
        """
        if self.has_model("ae_deep") and fallback_model is not None:
            # Reutiliza el StandardScaler del fallback model para mantener calibración
            session = self.sessions["ae_deep"]
            input_name = session.get_inputs()[0].name
            X_scaled = fallback_model.scaler.transform(features_68).astype(np.float32)
            outputs = session.run(None, {input_name: X_scaled})
            reconstructed = outputs[0]
            mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
            # Cálculo de Health Score calibrado
            max_mse = getattr(fallback_model, '_max_mse', 5.0)
            if max_mse <= 0:
                max_mse = 5.0
            anomaly_ratio = np.clip(mse / max_mse, 0.0, 1.0)
            health_scores = 100.0 * (1.0 - anomaly_ratio)
            is_anomaly = (health_scores < 50.0).astype(int)
            return is_anomaly, health_scores

        if fallback_model is not None:
            return fallback_model.predict_anomaly(features_68)

        raise RuntimeError("No se encontró sesión ONNX 'ae_deep' ni modelo de fallback nativo.")
