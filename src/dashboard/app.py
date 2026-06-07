import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Configuración de página
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    """Carga los datos procesados y resultados del torneo."""
    data = {}

    processed_path = "./data/processed/sensors_engineered.csv"
    if os.path.exists(processed_path):
        data['sensors'] = pd.read_csv(processed_path)

    results_dir = "./src/evaluation/results/"
    if os.path.exists(os.path.join(results_dir, "y_test.npy")):
        data['y_test'] = np.load(os.path.join(results_dir, "y_test.npy"))
        data['test_equipment'] = np.load(os.path.join(results_dir, "test_equipment_ids.npy"))
        data['test_rul'] = np.load(os.path.join(results_dir, "test_rul.npy"))
        data['ae_deep_health'] = np.load(os.path.join(results_dir, "ae_deep_health_scores.npy"))
        data['ae_deep_preds'] = np.load(os.path.join(results_dir, "ae_deep_predictions.npy"))
        data['ae_lstm_health'] = np.load(os.path.join(results_dir, "ae_lstm_health_scores.npy"))
        data['ae_lstm_preds'] = np.load(os.path.join(results_dir, "ae_lstm_predictions.npy"))
        data['gan_health'] = np.load(os.path.join(results_dir, "gan_health_scores.npy"))
        data['gan_preds'] = np.load(os.path.join(results_dir, "gan_predictions.npy"))
        data['xgb_preds'] = np.load(os.path.join(results_dir, "xgb_predictions.npy"))
        data['xgb_probs'] = np.load(os.path.join(results_dir, "xgb_probs.npy"))
        data['iso_preds'] = np.load(os.path.join(results_dir, "iso_predictions.npy"))
        data['iso_scores'] = np.load(os.path.join(results_dir, "iso_scores.npy"))

    tournament_path = os.path.join(results_dir, "maintenance_tournament.csv")
    if os.path.exists(tournament_path):
        data['tournament'] = pd.read_csv(tournament_path)

    importances_path = os.path.join(results_dir, "feature_importances.csv")
    if os.path.exists(importances_path):
        data['importances'] = pd.read_csv(importances_path)

    return data


def page_fleet_overview(data: dict):
    """Página 1: Visión General de la Flota."""
    st.header("🚛 Visión General de la Flota")

    if 'sensors' not in data:
        st.warning("⚠️ Datos no disponibles. Ejecuta `main_preprocessing.py` primero.")
        return

    df = data['sensors']

    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)
    n_equipment = df['equipment_id'].nunique()
    n_readings = len(df)
    n_normal = (df['machine_status'] == 0).sum()
    n_warning = (df['machine_status'] == 1).sum()
    n_critical = (df['machine_status'] == 2).sum()

    col1.metric("Equipos en Flota", f"{n_equipment}")
    col2.metric("Lecturas Totales", f"{n_readings:,}")
    col3.metric("⚠️ Alertas Warning", f"{n_warning:,}", f"{n_warning/n_readings*100:.1f}%")
    col4.metric("🔴 Alertas Critical", f"{n_critical:,}", f"{n_critical/n_readings*100:.1f}%")

    st.divider()

    # Distribución de estados
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribución de Estados")
        status_counts = df['machine_status'].value_counts().sort_index()
        labels = ['Normal', 'Warning', 'Critical']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=[status_counts.get(i, 0) for i in range(3)],
            marker=dict(colors=colors),
            hole=0.4,
        )])
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Sensores Clave (Promedio por Estado)")
        sensors = ['engine_temp', 'oil_pressure', 'vibration_level', 'fuel_consumption']
        means = df.groupby('machine_status')[sensors].mean()
        means.index = ['Normal', 'Warning', 'Critical']

        fig = go.Figure()
        for i, sensor in enumerate(sensors):
            fig.add_trace(go.Bar(
                name=sensor.replace('_', ' ').title(),
                x=['Normal', 'Warning', 'Critical'],
                y=means[sensor].values,
            ))
        fig.update_layout(
            barmode='group', template='plotly_dark', height=400,
            yaxis_title='Valor Promedio'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Timeline de un equipo específico
    st.divider()
    st.subheader("📈 Timeline de Sensores por Equipo")
    eq_ids = sorted(df['equipment_id'].unique())
    selected_eq = st.selectbox("Selecciona equipo:", eq_ids, format_func=lambda x: f"EQ-{x:03d}")

    eq_data = df[df['equipment_id'] == selected_eq].sort_values('cycle')
    sensor_choice = st.selectbox(
        "Sensor a visualizar:",
        ['engine_temp', 'oil_pressure', 'vibration_level', 'rpm',
         'fuel_consumption', 'coolant_temp', 'hydraulic_pressure']
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq_data['cycle'], y=eq_data[sensor_choice],
        mode='lines', name=sensor_choice,
        line=dict(color='#3498db', width=1.5),
    ))

    # Colorear fondo por estado
    for status, color, label in [(1, 'rgba(243,156,18,0.15)', 'Warning'),
                                  (2, 'rgba(231,76,60,0.25)', 'Critical')]:
        status_data = eq_data[eq_data['machine_status'] == status]
        if not status_data.empty:
            fig.add_vrect(
                x0=status_data['cycle'].min(), x1=status_data['cycle'].max(),
                fillcolor=color, line_width=0, annotation_text=label,
            )

    fig.update_layout(
        template='plotly_dark', height=400,
        xaxis_title='Ciclo de Operación',
        yaxis_title=sensor_choice.replace('_', ' ').title(),
    )
    st.plotly_chart(fig, use_container_width=True)


def page_health_monitoring(data: dict):
    """PAgina 2: Monitoreo de Salud (Health Score del Autoencoder)."""
    st.header("🩺 Monitoreo de Salud de Equipos")

    if 'ae_lstm_health' not in data:
        st.warning("⚠️ Health Scores no disponibles. Ejecuta `main_training.py`.")
        return

    ae_model_choice = st.radio("Modelo de Salud para RUL:", ["LSTM Autoencoder", "Deep Denoising Autoencoder", "GAN Anomaly Detector"], horizontal=True)
    if ae_model_choice == "LSTM Autoencoder":
        health_scores = data['ae_lstm_health']
    elif ae_model_choice == "GAN Anomaly Detector":
        health_scores = data['gan_health']
    else:
        health_scores = data['ae_deep_health']

    equipment_ids = data['test_equipment']
    rul = data['test_rul']

    # DataFrame de test con health scores
    test_df = pd.DataFrame({
        'equipment_id': equipment_ids,
        'rul': rul,
        'health_score': health_scores,
    })

    # KPIs de salud
    col1, col2, col3, col4 = st.columns(4)
    avg_health = health_scores.mean()
    min_health = health_scores.min()
    n_warning = (health_scores < 60).sum()
    n_critical = (health_scores < 30).sum()

    col1.metric("Health Score Promedio", f"{avg_health:.1f}")
    col2.metric("Health Score Mínimo", f"{min_health:.1f}")
    col3.metric("⚠️ Equipos en Warning", f"{n_warning:,}")
    col4.metric("🔴 Equipos en Critical", f"{n_critical:,}")

    st.divider()

    # Health Score vs RUL por equipo
    st.subheader("Health Score vs Remaining Useful Life")

    eq_ids = sorted(test_df['equipment_id'].unique())
    selected_eq = st.selectbox("Equipo:", eq_ids, format_func=lambda x: f"EQ-{x:03d}")

    eq_test = test_df[test_df['equipment_id'] == selected_eq].reset_index(drop=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=eq_test.index, y=eq_test['health_score'],
                   name='Health Score', line=dict(color='#2ecc71', width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=eq_test.index, y=eq_test['rul'],
                   name='RUL Real', line=dict(color='#3498db', width=2, dash='dot')),
        secondary_y=True,
    )

    # Líneas de threshold
    fig.add_hline(y=60, line_dash='dash', line_color='#f39c12',
                  annotation_text='Warning (60)', secondary_y=False)
    fig.add_hline(y=30, line_dash='dash', line_color='#e74c3c',
                  annotation_text='Critical (30)', secondary_y=False)

    fig.update_layout(template='plotly_dark', height=450)
    fig.update_yaxes(title_text="Health Score", secondary_y=False)
    fig.update_yaxes(title_text="RUL (ciclos)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # Distribución de Health Scores
    st.subheader("Distribución de Health Scores")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=health_scores, nbinsx=50, marker_color='#2ecc71', opacity=0.8,
    ))
    fig.add_vline(x=60, line_dash='dash', line_color='#f39c12', annotation_text='Warning')
    fig.add_vline(x=30, line_dash='dash', line_color='#e74c3c', annotation_text='Critical')
    fig.update_layout(template='plotly_dark', height=350,
                      xaxis_title='Health Score', yaxis_title='Frecuencia')
    st.plotly_chart(fig, use_container_width=True)


def page_model_tournament(data: dict):
    """Página 3: Torneo de Modelos."""
    st.header("🏆 Torneo de Modelos")

    if 'tournament' not in data:
        st.warning("⚠️ Resultados no disponibles. Ejecuta `main_evaluation.py`.")
        return

    tournament = data['tournament']

    # Tabla del torneo
    st.subheader("Tabla Comparativa")
    st.dataframe(tournament, use_container_width=True)

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Métricas de Clasificación")
        metric_cols = [c for c in tournament.columns if c in
                       ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']]
        if metric_cols:
            metrics_melt = tournament.melt(
                id_vars='model', value_vars=metric_cols,
                var_name='Métrica', value_name='Valor'
            )
            fig = px.bar(
                metrics_melt, x='Métrica', y='Valor', color='model',
                barmode='group', template='plotly_dark',
                color_discrete_sequence=['#3498db', '#e74c3c', '#f39c12'],
            )
            fig.update_layout(height=400, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Feature Importances
        if 'importances' in data:
            st.subheader("Top 15 Features (XGBoost)")
            imp = data['importances'].head(15)
            fig = px.bar(
                imp, x='importance', y='feature', orientation='h',
                template='plotly_dark', color='importance',
                color_continuous_scale='Viridis',
            )
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix del XGBoost
    if 'xgb_preds' in data:
        st.divider()
        st.subheader("Matriz de Confusión — XGBoost Multi-Class")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(data['y_test'], data['xgb_preds'], labels=[0, 1, 2])
        fig = px.imshow(
            cm, text_auto=True,
            labels=dict(x='Predicción', y='Real', color='Cantidad'),
            x=['Normal', 'Warning', 'Critical'],
            y=['Normal', 'Warning', 'Critical'],
            template='plotly_dark', color_continuous_scale='RdYlGn_r',
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)


def page_sensor_explorer(data: dict):
    """Página 4: Explorador de Sensores."""
    st.header("🔎 Explorador de Sensores")

    if 'sensors' not in data:
        st.warning("⚠️ Datos no disponibles.")
        return

    df = data['sensors']

    # Correlación entre sensores
    st.subheader("Correlación entre Sensores")
    sensor_cols = ['engine_temp', 'oil_pressure', 'vibration_level',
                   'rpm', 'fuel_consumption', 'coolant_temp', 'hydraulic_pressure']
    corr = df[sensor_cols].corr()
    fig = px.imshow(
        corr, text_auto='.2f',
        template='plotly_dark', color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Multi-sensor view por equipo
    st.divider()
    st.subheader("Vista Multi-Sensor por Equipo")
    eq_ids = sorted(df['equipment_id'].unique())
    selected_eq = st.selectbox("Equipo:", eq_ids, format_func=lambda x: f"EQ-{x:03d}", key='explorer_eq')
    eq_data = df[df['equipment_id'] == selected_eq].sort_values('cycle')

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=['Temperatura Motor (°C)', 'Vibración (mm/s)', 'Presión Aceite (PSI)'])

    fig.add_trace(go.Scatter(x=eq_data['cycle'], y=eq_data['engine_temp'],
                             line=dict(color='#e74c3c')), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq_data['cycle'], y=eq_data['vibration_level'],
                             line=dict(color='#f39c12')), row=2, col=1)
    fig.add_trace(go.Scatter(x=eq_data['cycle'], y=eq_data['oil_pressure'],
                             line=dict(color='#3498db')), row=3, col=1)

    fig.update_layout(template='plotly_dark', height=700, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Estadísticas descriptivas
    st.divider()
    st.subheader("Estadísticas Descriptivas por Estado")
    status_filter = st.selectbox("Estado:", ['Todos', 'Normal', 'Warning', 'Critical'])
    if status_filter == 'Normal':
        filtered = df[df['machine_status'] == 0]
    elif status_filter == 'Warning':
        filtered = df[df['machine_status'] == 1]
    elif status_filter == 'Critical':
        filtered = df[df['machine_status'] == 2]
    else:
        filtered = df

    st.dataframe(filtered[sensor_cols].describe().round(2), use_container_width=True)


# =============================================================
# MAIN APP
# =============================================================
def main():
    st.sidebar.title("🚛 Predictive Maintenance")
    st.sidebar.markdown("**Sistema de Mantenimiento Predictivo**")
    st.sidebar.markdown("Detección de anomalías en\nsensores de maquinaria minera")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navegación",
        ["🚛 Visión de Flota", "💚 Monitoreo de Salud",
         "🏆 Torneo de Modelos", "🔎 Explorador de Sensores"]
    )

    data = load_data()

    if page == "🚛 Visión de Flota":
        page_fleet_overview(data)
    elif page == "💚 Monitoreo de Salud":
        page_health_monitoring(data)
    elif page == "🏆 Torneo de Modelos":
        page_model_tournament(data)
    elif page == "🔎 Explorador de Sensores":
        page_sensor_explorer(data)

    st.sidebar.divider()
    st.sidebar.caption("Predictive Maintenance System v1.0")
    st.sidebar.caption("NVIDIA Anomaly Detection Certified")


if __name__ == "__main__":
    main()
