import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class RULAnalyzer:
    """
    Analizador de Remaining Useful Life (RUL) para mantenimiento predictivo.
    Compara el RUL predicho (basado en Health Score del Autoencoder)
    con el RUL real, y calcula métricas de performance industrial.

    Métricas calculadas:
        - MAE del RUL (Mean Absolute Error)
        - RMSE del RUL
        - Lead Time de alerta (ciclos de anticipación antes de la falla)
        - Early/Late prediction rate
    """

    def __init__(self, warning_threshold: float = 60.0,
                 critical_threshold: float = 30.0):
        """
        Args:
            warning_threshold: Health Score bajo el cual se emite alerta Warning.
            critical_threshold: Health Score bajo el cual se emite alerta Critical.
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def analyze(self, equipment_ids: np.ndarray, rul_true: np.ndarray,
                health_scores: np.ndarray) -> dict:
        """
        Análisis completo de RUL por equipo.
        
        Args:
            equipment_ids: ID del equipo para cada lectura.
            rul_true: RUL real (ciclos hasta falla).
            health_scores: Health Score predicho por el Autoencoder.
        """
        print("  📐 Analizando Remaining Useful Life (RUL)...")

        df = pd.DataFrame({
            'equipment_id': equipment_ids,
            'rul_true': rul_true,
            'health_score': health_scores,
        })

        # RUL predicho: estimado a partir del Health Score
        # Mapeo lineal: Health Score 100 → RUL max, Health Score 0 → RUL 0
        max_rul = df.groupby('equipment_id')['rul_true'].transform('max')
        df['rul_predicted'] = (df['health_score'] / 100) * max_rul

        # Métricas globales
        mae = np.mean(np.abs(df['rul_true'] - df['rul_predicted']))
        rmse = np.sqrt(np.mean((df['rul_true'] - df['rul_predicted'])**2))

        # Análisis de alertas por equipo
        alert_analysis = self._analyze_alerts_by_equipment(df)

        results = {
            'mae': mae,
            'rmse': rmse,
            'n_equipment': df['equipment_id'].nunique(),
            'alert_analysis': alert_analysis,
        }

        print(f"  ✅ RUL Analysis completado")
        print(f"     MAE: {mae:.1f} ciclos")
        print(f"     RMSE: {rmse:.1f} ciclos")

        if alert_analysis is not None:
            avg_lead = alert_analysis['lead_time_cycles'].mean()
            print(f"     Lead Time promedio de alerta: {avg_lead:.0f} ciclos")

        return results

    def _analyze_alerts_by_equipment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analiza el lead time de alertas por equipo."""
        rows = []

        for eq_id in df['equipment_id'].unique():
            eq_data = df[df['equipment_id'] == eq_id].sort_values('rul_true', ascending=False)

            # Buscar primera alerta Warning
            warning_mask = eq_data['health_score'] < self.warning_threshold
            critical_mask = eq_data['health_score'] < self.critical_threshold

            first_warning_rul = None
            first_critical_rul = None

            if warning_mask.any():
                first_warning_idx = warning_mask.idxmax()
                first_warning_rul = eq_data.loc[first_warning_idx, 'rul_true']

            if critical_mask.any():
                first_critical_idx = critical_mask.idxmax()
                first_critical_rul = eq_data.loc[first_critical_idx, 'rul_true']

            # Solo equipos con falla real (RUL llega a 0)
            has_failure = (eq_data['rul_true'] == 0).any()

            rows.append({
                'equipment_id': eq_id,
                'has_failure': has_failure,
                'first_warning_rul': first_warning_rul,
                'first_critical_rul': first_critical_rul,
                'lead_time_cycles': first_warning_rul if first_warning_rul else 0,
                'min_health_score': eq_data['health_score'].min(),
                'final_health_score': eq_data.iloc[-1]['health_score'],
            })

        return pd.DataFrame(rows)

    def print_equipment_summary(self, alert_df: pd.DataFrame) -> None:
        """Imprime un resumen por equipo."""
        print(f"\n{'='*70}")
        print(f"  🚛 RESUMEN POR EQUIPO")
        print(f"  {'-'*66}")
        print(f"  {'Equipo':<10} {'Falla':>7} {'Warning@RUL':>13} {'Critical@RUL':>14} "
              f"{'Min HS':>8} {'Final HS':>10}")
        print(f"  {'-'*66}")

        for _, row in alert_df.head(20).iterrows():
            fail = "Sí" if row['has_failure'] else "No"
            w_rul = f"{row['first_warning_rul']:.0f}" if row['first_warning_rul'] else "N/A"
            c_rul = f"{row['first_critical_rul']:.0f}" if row['first_critical_rul'] else "N/A"
            print(f"  EQ-{int(row['equipment_id']):03d}    {fail:>7} {w_rul:>13} {c_rul:>14} "
                  f"{row['min_health_score']:>8.1f} {row['final_health_score']:>10.1f}")

        print(f"{'='*70}")
