import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def create_health_timeline(equipment_id: int, cycles: np.ndarray,
                            health_scores: np.ndarray, rul: np.ndarray) -> go.Figure:
    """
    Genera un gráfico de timeline con Health Score y RUL para un equipo.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=cycles, y=health_scores,
                   name='Health Score', fill='tozeroy',
                   line=dict(color='#2ecc71', width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=cycles, y=rul,
                   name='RUL', line=dict(color='#3498db', width=2, dash='dot')),
        secondary_y=True,
    )

    fig.add_hline(y=60, line_dash='dash', line_color='#f39c12',
                  annotation_text='Warning', secondary_y=False)
    fig.add_hline(y=30, line_dash='dash', line_color='#e74c3c',
                  annotation_text='Critical', secondary_y=False)

    fig.update_layout(
        title=f'EQ-{equipment_id:03d} — Health Score vs RUL',
        template='plotly_dark', height=400,
    )
    fig.update_yaxes(title_text="Health Score (0-100)", secondary_y=False)
    fig.update_yaxes(title_text="RUL (ciclos)", secondary_y=True)
    return fig


def create_sensor_comparison(df: pd.DataFrame, equipment_id: int,
                              sensors: list) -> go.Figure:
    """Genera gráficos de múltiples sensores para un equipo."""
    eq_data = df[df['equipment_id'] == equipment_id].sort_values('cycle')
    n_sensors = len(sensors)

    fig = make_subplots(
        rows=n_sensors, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[s.replace('_', ' ').title() for s in sensors]
    )

    colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6', '#1abc9c', '#e67e22']

    for i, sensor in enumerate(sensors):
        fig.add_trace(
            go.Scatter(x=eq_data['cycle'], y=eq_data[sensor],
                       line=dict(color=colors[i % len(colors)], width=1.5),
                       showlegend=False),
            row=i + 1, col=1
        )

    fig.update_layout(
        template='plotly_dark',
        height=200 * n_sensors,
        title=f'EQ-{equipment_id:03d} — Vista Multi-Sensor',
    )
    return fig


def create_fleet_heatmap(df: pd.DataFrame, sensor: str) -> go.Figure:
    """Genera un heatmap de un sensor para toda la flota."""
    pivot = df.pivot_table(
        values=sensor, index='equipment_id', columns='cycle', aggfunc='mean'
    )

    fig = px.imshow(
        pivot, template='plotly_dark',
        color_continuous_scale='RdYlGn_r',
        labels=dict(x='Ciclo', y='Equipo', color=sensor),
        title=f'Heatmap de Flota — {sensor.replace("_", " ").title()}',
    )
    fig.update_layout(height=600)
    return fig
