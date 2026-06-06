import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings

warnings.filterwarnings("ignore")


class MaintenanceMetricsEngine:
    """
    Motor de evaluación comparativa para modelos de mantenimiento predictivo.
    Calcula métricas estándar y métricas industriales específicas.

    Métricas calculadas:
        - Accuracy, Precision, Recall, F1-Score (macro y por clase)
        - Confusion Matrix (3x3: Normal/Warning/Critical)
        - Lead Time de alerta (cuántos ciclos antes de la falla se detecta)
        - False Alarm Rate
        - Costo de downtime evitado vs costo de mantenimiento preventivo
    """

    def __init__(self, cost_downtime_hour: float = 5000.0,
                 cost_maintenance: float = 2000.0):
        """
        Args:
            cost_downtime_hour: Costo por hora de parada no planificada (USD).
            cost_maintenance: Costo de una intervención preventiva (USD).
        """
        self.cost_downtime = cost_downtime_hour
        self.cost_maintenance = cost_maintenance
        self.results = {}

    def evaluate_model(self, model_name: str, y_true: np.ndarray,
                       y_pred: np.ndarray, is_multiclass: bool = True) -> dict:
        """Evalúa un modelo y almacena los resultados."""
        if is_multiclass:
            metrics = self._evaluate_multiclass(model_name, y_true, y_pred)
        else:
            metrics = self._evaluate_binary(model_name, y_true, y_pred)

        self.results[model_name] = metrics
        return metrics

    def _evaluate_multiclass(self, model_name: str, y_true: np.ndarray,
                              y_pred: np.ndarray) -> dict:
        """Evaluación para clasificación 3 clases."""
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        # Métricas por clase
        for cls, label in enumerate(['normal', 'warning', 'critical']):
            y_cls = (y_true == cls).astype(int)
            p_cls = (y_pred == cls).astype(int)
            metrics[f'precision_{label}'] = precision_score(y_cls, p_cls, zero_division=0)
            metrics[f'recall_{label}'] = recall_score(y_cls, p_cls, zero_division=0)
            metrics[f'f1_{label}'] = f1_score(y_cls, p_cls, zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        metrics['confusion_matrix'] = cm

        # Métricas industriales
        metrics.update(self._calculate_industrial_metrics(y_true, y_pred))

        return metrics

    def _evaluate_binary(self, model_name: str, y_true: np.ndarray,
                          y_pred: np.ndarray) -> dict:
        """Evaluación binaria (anomalía sí/no)."""
        # Convertir multi-clase a binario: 0=Normal, 1=Anómalo (Warning+Critical)
        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int) if y_pred.max() > 1 else y_pred

        cm = confusion_matrix(y_true_bin, y_pred_bin)
        tn, fp, fn, tp = cm.ravel()

        return {
            'model': model_name,
            'accuracy': accuracy_score(y_true_bin, y_pred_bin),
            'precision_macro': precision_score(y_true_bin, y_pred_bin, zero_division=0),
            'recall_macro': recall_score(y_true_bin, y_pred_bin, zero_division=0),
            'f1_macro': f1_score(y_true_bin, y_pred_bin, zero_division=0),
            'f1_weighted': f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
        }

    def _calculate_industrial_metrics(self, y_true: np.ndarray,
                                       y_pred: np.ndarray) -> dict:
        """Calcula métricas específicas de la industria minera."""
        # False Alarm Rate: predicciones de falla cuando todo está normal
        normal_mask = y_true == 0
        false_alarms = np.sum(y_pred[normal_mask] > 0)
        far = false_alarms / max(np.sum(normal_mask), 1)

        # Detection Rate: fallas críticas detectadas
        critical_mask = y_true == 2
        detected_critical = np.sum(y_pred[critical_mask] >= 1)  # Al menos Warning
        detection_rate = detected_critical / max(np.sum(critical_mask), 1)

        # Costo estimado
        missed_failures = np.sum(y_pred[critical_mask] == 0)
        preventive_actions = np.sum(y_pred > 0)
        estimated_cost = (
            missed_failures * self.cost_downtime * 8 +  # 8 horas de downtime
            preventive_actions * self.cost_maintenance
        )

        return {
            'false_alarm_rate': far,
            'critical_detection_rate': detection_rate,
            'estimated_cost': estimated_cost,
        }

    def print_model_report(self, model_name: str) -> None:
        """Imprime un reporte detallado para un modelo específico."""
        metrics = self.results.get(model_name)
        if metrics is None:
            print(f"  ⚠️ Modelo {model_name} no evaluado.")
            return

        print(f"\n{'='*60}")
        print(f"  📊 Reporte: {model_name}")
        print(f"{'='*60}")
        print(f"  Accuracy:        {metrics['accuracy']:.4f}")
        print(f"  Precision (M):   {metrics['precision_macro']:.4f}")
        print(f"  Recall (M):      {metrics['recall_macro']:.4f}")
        print(f"  F1-Score (M):    {metrics['f1_macro']:.4f}")

        if 'false_alarm_rate' in metrics:
            print(f"\n  🏭 Métricas Industriales:")
            print(f"     False Alarm Rate:        {metrics['false_alarm_rate']:.2%}")
            print(f"     Critical Detection Rate: {metrics['critical_detection_rate']:.2%}")
            print(f"     Costo Estimado:          ${metrics['estimated_cost']:,.0f}")

        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            print(f"\n  Confusion Matrix (N/W/C):")
            for i, label in enumerate(['Normal  ', 'Warning ', 'Critical']):
                print(f"    {label} → [{cm[i][0]:5d} {cm[i][1]:5d} {cm[i][2]:5d}]")

    def print_tournament_results(self) -> None:
        """Imprime los resultados del torneo de modelos."""
        if not self.results:
            return

        print("\n" + "=" * 85)
        print("  🏆 TORNEO DE MODELOS — MANTENIMIENTO PREDICTIVO")
        print("  " + "-" * 81)
        print(f"  {'Modelo':<25} {'Acc':>7} {'Prec(M)':>9} {'Rec(M)':>8} "
              f"{'F1(M)':>7} {'FAR':>8} {'Det.Crit':>10}")
        print("  " + "-" * 81)

        sorted_models = sorted(self.results.items(),
                                key=lambda x: x[1].get('f1_macro', 0), reverse=True)

        for name, m in sorted_models:
            far = f"{m.get('false_alarm_rate', 0):.2%}" if 'false_alarm_rate' in m else 'N/A'
            det = f"{m.get('critical_detection_rate', 0):.2%}" if 'critical_detection_rate' in m else 'N/A'
            print(f"  {name:<25} {m['accuracy']:>7.4f} {m['precision_macro']:>9.4f} "
                  f"{m['recall_macro']:>8.4f} {m['f1_macro']:>7.4f} {far:>8} {det:>10}")

        print("=" * 85)
        winner = sorted_models[0]
        print(f"  🥇 Ganador: {winner[0]} (Macro F1: {winner[1]['f1_macro']:.4f})")

    def save_results(self, output_path: str) -> None:
        """Guarda los resultados del torneo en CSV."""
        rows = []
        for name, m in self.results.items():
            row = {k: v for k, v in m.items() if k != 'confusion_matrix'}
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"  💾 Resultados guardados en: {output_path}")
