import os
import numpy as np
import pandas as pd
from preprocessing.feature_engineer import SensorFeatureEngineer
from models.autoencoder_deep import MaintenanceDeepAutoencoder
from models.autoencoder_lstm import MaintenanceLSTMAutoencoder
from models.xgb_predictor import MaintenanceXGBoostPredictor
from models.isolation_forest import MaintenanceIsolationForest
import warnings

warnings.filterwarnings("ignore")


def run_training_pipeline():
    """
    Bloque 2: Pipeline de Entrenamiento de Modelos.
    Carga el dataset procesado, divide en train/test por equipo,
    y entrena los tres modelos de mantenimiento predictivo.
    """
    print("🚀 Iniciando Pipeline de Entrenamiento (Bloque 2)...")

    # --- PASO 1: Carga del Dataset Procesado ---
    print("\n--- PASO 1: Cargando Dataset Procesado ---")
    processed_path = "./data/processed/sensors_engineered.csv"
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"❌ Dataset no encontrado en {processed_path}\n"
            "   Ejecuta primero: python src/main_preprocessing.py"
        )

    df = pd.read_csv(processed_path)
    print(f"  📊 Dataset cargado: {len(df):,} registros, {df['equipment_id'].nunique()} equipos")

    # --- PASO 2: Preparación de Features ---
    print("\n--- PASO 2: Preparación de Features ---")
    engineer = SensorFeatureEngineer()
    feature_cols = engineer.get_feature_columns()
    available_features = [c for c in feature_cols if c in df.columns]
    print(f"  📐 Features seleccionadas: {len(available_features)}")

    # División por equipo: 80% equipos para train, 20% para test
    equipment_ids = df['equipment_id'].unique()
    np.random.seed(42)
    np.random.shuffle(equipment_ids)
    split_idx = int(len(equipment_ids) * 0.80)
    train_equipment = equipment_ids[:split_idx]
    test_equipment = equipment_ids[split_idx:]

    train_mask = df['equipment_id'].isin(train_equipment)
    test_mask = df['equipment_id'].isin(test_equipment)

    X_train = df.loc[train_mask, available_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = df.loc[train_mask, 'machine_status'].values
    X_test = df.loc[test_mask, available_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_test = df.loc[test_mask, 'machine_status'].values

    print(f"  📦 Train: {len(X_train):,} lecturas ({len(train_equipment)} equipos)")
    print(f"  📦 Test: {len(X_test):,} lecturas ({len(test_equipment)} equipos)")

    os.makedirs("./src/evaluation/results", exist_ok=True)

    # --- MODELO 1A: Deep Denoising Autoencoder (Health Score) ---
    print("\n--- MODELO 1A: Deep Denoising Autoencoder (Health Score) ---")
    autoencoder_deep = MaintenanceDeepAutoencoder(encoding_dim=16, epochs=100, batch_size=128)

    # Entrenar solo con datos normales
    X_train_normal = X_train[y_train == 0]
    ae_info = autoencoder_deep.fit(X_train_normal)

    ae_preds, ae_health = autoencoder_deep.predict_anomaly(X_test, threshold=50.0)
    np.save("./src/evaluation/results/ae_deep_health_scores.npy", ae_health)
    np.save("./src/evaluation/results/ae_deep_predictions.npy", ae_preds)
    print(f"  💾 Health Scores del Deep Autoencoder guardados")

    # --- MODELO 1B: LSTM Autoencoder (Health Score) ---
    print("\n--- MODELO 1B: LSTM Autoencoder (Health Score) ---")
    autoencoder_lstm = MaintenanceLSTMAutoencoder(encoding_dim=16, epochs=100, batch_size=128)

    ae_lstm_info = autoencoder_lstm.fit(X_train_normal)

    ae_lstm_preds, ae_lstm_health = autoencoder_lstm.predict_anomaly(X_test, threshold=50.0)
    np.save("./src/evaluation/results/ae_lstm_health_scores.npy", ae_lstm_health)
    np.save("./src/evaluation/results/ae_lstm_predictions.npy", ae_lstm_preds)
    print(f"  💾 Health Scores del LSTM Autoencoder guardados")

    # --- MODELO 2: XGBoost Multi-Class ---
    print("\n--- MODELO 2: XGBoost Multi-Class ---")
    xgb_predictor = MaintenanceXGBoostPredictor(n_splits=5, purge_size=30, embargo_size=10)

    best_params = xgb_predictor.find_best_params(X_train, y_train)
    xgb_predictor.train(X_train, y_train, best_params)

    xgb_preds, xgb_probs = xgb_predictor.predict(X_test)
    np.save("./src/evaluation/results/xgb_predictions.npy", xgb_preds)
    np.save("./src/evaluation/results/xgb_probs.npy", xgb_probs)

    importances = xgb_predictor.get_feature_importances(available_features)
    importances.to_csv("./src/evaluation/results/feature_importances.csv", index=False)
    print(f"  💾 Predicciones e importancias del XGBoost guardadas")

    # --- MODELO 3: Isolation Forest ---
    print("\n--- MODELO 3: Isolation Forest (Baseline) ---")
    iso_forest = MaintenanceIsolationForest(contamination=0.10, n_estimators=200)
    iso_forest.fit(X_train)

    iso_preds, iso_scores = iso_forest.predict(X_test)
    np.save("./src/evaluation/results/iso_predictions.npy", iso_preds)
    np.save("./src/evaluation/results/iso_scores.npy", iso_scores)
    print(f"  💾 Predicciones del Isolation Forest guardadas")

    # Guardar labels y metadata de test
    np.save("./src/evaluation/results/y_test.npy", y_test)
    np.save("./src/evaluation/results/test_equipment_ids.npy",
            df.loc[test_mask, 'equipment_id'].values)
    np.save("./src/evaluation/results/test_rul.npy",
            df.loc[test_mask, 'rul'].values)

    print(f"\n✅ ¡Pipeline de entrenamiento completado!")
    print(f"📁 Artefactos guardados en: ./src/evaluation/results/")


if __name__ == "__main__":
    run_training_pipeline()
