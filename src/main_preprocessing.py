import os
import warnings
from preprocessing.data_synthesizer import SensorDataSynthesizer
from preprocessing.data_loader import SensorDataLoader
from preprocessing.feature_engineer import SensorFeatureEngineer

warnings.filterwarnings("ignore")


def run_preprocessing_pipeline():
    """
    Bloque 1: Pipeline de Preprocesamiento de Datos de Sensores.
    Genera datos sintéticos de equipos mineros (si no existen),
    los carga, valida, y aplica Feature Engineering temporal.
    """
    print("🚀 Iniciando Pipeline de Preprocesamiento (Bloque 1)...")

    # --- PASO 1: Generación de Datos Sintéticos ---
    raw_path = "./data/raw/sensor_readings.csv"
    if not os.path.exists(raw_path):
        print("\n--- PASO 1: Generación de Datos Sintéticos de Sensores ---")
        os.makedirs("./data/raw", exist_ok=True)
        synthesizer = SensorDataSynthesizer(
            n_equipment=50,
            cycles_per_equipment=300,
            failure_rate=0.30,
        )
        synthesizer.generate(output_path=raw_path)
    else:
        print(f"\n--- PASO 1: Datos ya existen en {raw_path} (skip generación) ---")

    # --- PASO 2: Carga y Validación ---
    print("\n--- PASO 2: Carga y Validación de Datos de Sensores ---")
    loader = SensorDataLoader(raw_data_path="./data/raw/")
    df = loader.load_sensor_data()

    # --- PASO 3: Feature Engineering ---
    print("\n--- PASO 3: Feature Engineering de Series Temporales ---")
    engineer = SensorFeatureEngineer(windows=[5, 10, 20])
    df = engineer.engineer_features(df)

    # --- PASO 4: Guardado ---
    print("\n--- PASO 4: Guardando Dataset Procesado ---")
    os.makedirs("./data/processed", exist_ok=True)
    output_path = "./data/processed/sensors_engineered.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ ¡Pipeline de preprocesamiento completado!")
    print(f"📁 Dataset enriquecido guardado en: {output_path}")


if __name__ == "__main__":
    run_preprocessing_pipeline()
