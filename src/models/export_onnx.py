"""
Utilidad de Exportación de Modelos a formato ONNX (.onnx).
Convierte modelos entrenados (XGBoost y Keras Autoencoder) para ejecución
de ultra alta velocidad en ONNX Runtime.
"""

import os
import sys
import logging
import joblib
import numpy as np

logger = logging.getLogger("export_onnx")


def export_models_to_onnx(models_dir: str = "./models/saved_models") -> None:
    """Intenta convertir los modelos entrenados presentes a ONNX."""
    print("🚀 Iniciando exportación de modelos a ONNX...")

    xgb_path = os.path.join(models_dir, "xgb.pkl")
    ae_deep_path = os.path.join(models_dir, "ae_deep.pkl")

    # 1. Exportación de XGBoost
    if os.path.exists(xgb_path):
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            xgb_state = joblib.load(xgb_path)
            xgb_model = xgb_state.get('model')
            if xgb_model is not None:
                initial_type = [('float_input', FloatTensorType([None, 68]))]
                onx = convert_sklearn(xgb_model, initial_types=initial_type, target_opset=13)
                out_xgb = os.path.join(models_dir, "xgb.onnx")
                with open(out_xgb, "wb") as f:
                    f.write(onx.SerializeToString())
                print(f"  ✅ XGBoost exportado exitosamente a: {out_xgb}")
        except Exception as e:
            print(f"  ℹ️ Exportación de XGBoost a ONNX omitida o requiere skl2onnx ({e}). Operando con fallback nativo.")

    # 2. Exportación de Deep Autoencoder
    if os.path.exists(ae_deep_path):
        try:
            import tf2onnx
            from tensorflow import keras
            keras_path = ae_deep_path.replace(".pkl", ".keras")
            if os.path.exists(keras_path):
                model = keras.models.load_model(keras_path)
                out_ae = os.path.join(models_dir, "ae_deep.onnx")
                # spec: input_dim 68
                import tensorflow as tf
                spec = (tf.TensorSpec((None, 68), tf.float32, name="float_input"),)
                tf2onnx.convert.from_keras(model, input_signature=spec, output_path=out_ae)
                print(f"  ✅ Deep Autoencoder exportado exitosamente a: {out_ae}")
        except Exception as e:
            print(f"  ℹ️ Exportación de Deep Autoencoder a ONNX omitida o requiere tf2onnx ({e}). Operando con fallback nativo.")

    print("🏁 Proceso de exportación finalizado.")


if __name__ == "__main__":
    export_models_to_onnx()
