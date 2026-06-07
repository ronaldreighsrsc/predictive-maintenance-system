import os
import numpy as np
from evaluation.metrics_engine import MaintenanceMetricsEngine
from evaluation.rul_analyzer import RULAnalyzer
import warnings

warnings.filterwarnings("ignore")


def run_evaluation_pipeline():
    """
    Bloque 3: Pipeline de Evaluación Comparativa.
    Carga las predicciones de los tres modelos, ejecuta el torneo
    de métricas con análisis industrial, y realiza el análisis de RUL.
    """
    print("🚀 Iniciando Pipeline de Evaluación (Bloque 3)...")

    results_dir = "./src/evaluation/results/"

    # --- PASO 1: Cargar Predicciones ---
    print("\n--- PASO 1: Cargando Predicciones ---")
    y_test = np.load(os.path.join(results_dir, "y_test.npy"))
    test_equipment = np.load(os.path.join(results_dir, "test_equipment_ids.npy"))
    test_rul = np.load(os.path.join(results_dir, "test_rul.npy"))

    n_normal = (y_test == 0).sum()
    n_warning = (y_test == 1).sum()
    n_critical = (y_test == 2).sum()
    print(f"  📊 Test: {len(y_test):,} lecturas")
    print(f"     Normal: {n_normal:,} | Warning: {n_warning:,} | Critical: {n_critical:,}")

    # --- PASO 2: Evaluación de Modelos ---
    print("\n--- PASO 2: Evaluación de Modelos ---")
    engine = MaintenanceMetricsEngine(
        cost_downtime_hour=5000.0,
        cost_maintenance=2000.0,
    )

    # XGBoost (multi-clase directo)
    xgb_preds = np.load(os.path.join(results_dir, "xgb_predictions.npy"))
    engine.evaluate_model('XGBoost Multi-Class', y_test, xgb_preds, is_multiclass=True)
    engine.print_model_report('XGBoost Multi-Class')

    # Deep Denoising Autoencoder (binario: anomalía sí/no)
    ae_deep_preds = np.load(os.path.join(results_dir, "ae_deep_predictions.npy"))
    engine.evaluate_model('Deep Denoising Autoencoder', y_test, ae_deep_preds, is_multiclass=False)
    engine.print_model_report('Deep Denoising Autoencoder')

    # LSTM Autoencoder (binario)
    ae_lstm_preds = np.load(os.path.join(results_dir, "ae_lstm_predictions.npy"))
    engine.evaluate_model('LSTM Autoencoder', y_test, ae_lstm_preds, is_multiclass=False)
    engine.print_model_report('LSTM Autoencoder')

    # Isolation Forest (binario)
    iso_preds = np.load(os.path.join(results_dir, "iso_predictions.npy"))
    engine.evaluate_model('Isolation Forest', y_test, iso_preds, is_multiclass=False)
    engine.print_model_report('Isolation Forest')

    # --- PASO 3: Análisis de RUL ---
    print("\n\n--- PASO 3: Análisis de Remaining Useful Life (RUL) ---")
    ae_deep_health = np.load(os.path.join(results_dir, "ae_deep_health_scores.npy"))
    ae_lstm_health = np.load(os.path.join(results_dir, "ae_lstm_health_scores.npy"))

    rul_analyzer = RULAnalyzer(warning_threshold=60.0, critical_threshold=30.0)
    
    print("\n--- RUL Analysis: Deep Denoising Autoencoder ---")
    rul_results_deep = rul_analyzer.analyze(test_equipment, test_rul, ae_deep_health)
    
    print("\n--- RUL Analysis: LSTM Autoencoder ---")
    rul_results_lstm = rul_analyzer.analyze(test_equipment, test_rul, ae_lstm_health)

    if rul_results_lstm['alert_analysis'] is not None:
        rul_analyzer.print_equipment_summary(rul_results_lstm['alert_analysis'])

    # --- PASO 4: Resultados del Torneo ---
    engine.print_tournament_results()
    engine.save_results(os.path.join(results_dir, "maintenance_tournament.csv"))

    print(f"\n✅ ¡Pipeline de evaluación completado!")


if __name__ == "__main__":
    run_evaluation_pipeline()
