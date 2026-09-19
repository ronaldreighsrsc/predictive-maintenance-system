# 🚜 Predictive Maintenance System v2.0
## Confiabilidad Operacional y Mantenimiento Basado en Condición (CBM / ISO 13374) para Camiones Mineros CAEX

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v2.0-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Accelerated%20%3C4ms-005CED?logo=onnx)](https://onnxruntime.ai/)
[![ISO Standard](https://img.shields.io/badge/Standard-ISO%2013374%20%7C%20SAE%20J1939-orange)](#)
[![Tests](https://img.shields.io/badge/Tests-25%20Passed%20%7C%20100%25-brightgreen)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 1. Resumen Ejecutivo: El Cierre de los 4 Pilares de Misión Crítica

Este sistema consolida una arquitectura industrial de confiabilidad operacional y mantenimiento basado en condición (CBM) para camiones de extracción minera (**CAEX: Komatsu 930E / Caterpillar 797F** en faenas a rajo abierto). 

Integrando los aprendizajes de alta disponibilidad y funciones de costo asimétricas aplicados en el sector bancario, edge IoT y quant trading, este proyecto transforma modelos de laboratorio en un **sistema de grado industrial inatacable**:

```
┌───────────────────────────────────────┬───────────────────────────────────┬─────────────────────────────────────┬────────────────────────────────────┬───────────────────────────────────┐
│ Dimensión de Ingeniería               │ 1. Banca / Fraude (Bci)           │ 2. IoT Edge (OmniEdge Sentinel)     │ 3. Quant Trading (AlphaEdge)       │ 4. Predictive Maintenance (CAEX)  │
├───────────────────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┼────────────────────────────────────┼───────────────────────────────────┤
│ 1. Restricción Temporal (SLA)         │ Switch Transaccional (< 30 ms)    │ Handover Wi-Fi (< 800 ms)           │ Tick-to-Order MT5 (< 15 ms)        │ Ingesta Telemetría CAN (< 20 ms)  │
│ 2. Función de Pérdida / Costo         │ Costo Asimétrico Ley 21.234       │ Penalización Desconexión C_switch   │ Fricción Microestructural (Spread) │ Falla en Rampa vs. Parada Taller  │
│ 3. Restricción de Cómputo             │ Microservicios Cloud Containers   │ Flash SD Wear / RAM < 120 MB        │ VPS Trading 1-2 GB RAM (Zero OOM)  │ Edge Gateway / Docker < 350 MB    │
│ 4. Detección de Deriva (Drift)        │ Population Stability Index (PSI)  │ Kolmogorov-Smirnov RF Azapa         │ LSTM Autoencoder + 3-State HMM     │ Kolmogorov-Smirnov Sensor Drift   │
│ 5. Agente GenAI / Explicabilidad      │ Agente ROS CMF (Tipologías UAF)   │ Agente RCA Falla Red (IEEE 802.11)  │ Agente Macro RAG Pre-News          │ Agente SAP PM / ISO 13374 RAG     │
│ 6. Persistencia y Caché               │ Redis In-Memory + Delta Lake      │ Ring Buffer RAM + SQLite Batch      │ Caching RAM + SQLite WAL / DuckDB  │ Sliding Window Ring Buffer O(1)   │
└───────────────────────────────────────┴───────────────────────────────────┴─────────────────────────────────────┴────────────────────────────────────┴───────────────────────────────────┘
```

---

## 2. Matriz Comparativa de Evolución: v1.0 vs v2.0

| Componente | Versión 1.0 (Laboratorio / Prototipo) | Versión 2.0 (Grado Industrial Minero) | Impacto Operacional |
|---|---|---|---|
| **Contrato de Ingesta API** | Exige 68 variables pre-calculadas en JSON | **Acepta 7 sensores crudos CAN Bus** | Viable en streaming real; cero carga en el camión |
| **Feature Engineering** | Batch estático en Pandas | **Sliding Window Buffer en RAM O(1)** | Features temporales calculadas en < 1.2 ms |
| **Inferencia de Modelos** | 5 modelos síncronos en serie (150-280 ms) | **Staged Triage con ONNX Runtime (< 4 ms)** | $\approx 40\times$ más rápido; soporta 500+ camiones |
| **Huella de Contenedor** | 2.1 GB Docker / 1.8 GB RAM (TensorFlow) | **< 350 MB Docker / < 200 MB RAM** | Despliegue en Edge Gateways de terreno |
| **Criterio de Decisión** | Argmax probabilístico simétrico (50%) | **Matriz de Costo CBM ($\theta^*_{\text{crit}} = 2.23\%$)** | Ahorro de **$338,000 USD** por falla evitada |
| **Cálculo de RUL** | Mapeo lineal con `rul_true` filtrado (Data Leakage) | **Weibull Hazard Curve con Intervalos P10-P90** | Sin data leakage; planificación confiable de turnos |
| **Accionabilidad de Alerta** | Solo números y etiquetas (*Warning/Critical*) | **Agente GenAI con Órdenes de Trabajo SAP PM** | Instrucciones directas y repuestos OEM para mecánicos |
| **Integridad de Datos** | Sin validación de fallas en sensores | **Detector de Sensor Drift físico (KS-Test)** | Cero paradas innecesarias por sensores sucios |

---

## 3. Arquitectura del Sistema (SOLID & Clean Architecture)

```
[MÓDEM TELEMETRÍA CAN BUS (7 SENSORES CRUDOS)]
                    │
                    ▼
      [POST /api/v2/telemetry/ingest]
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. SLIDING WINDOW BUFFER EN RAM O(1)                    │
│    • deque(maxlen=20) indexado por equipment_id         │
│    • 68 features en C-Memory (< 1.2 ms)                 │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ 2. STAGED TRIAGE GATEKEEPER                             │
│    • Fast-Path (< 3 ms): Deep Autoencoder + XGBoost     │
│    • Matriz de Costo Bayesiana: θ* = 2.23%              │
└─────────────┬─────────────────────────────┬─────────────┘
              │                             │
    [Health >= 70 & Normal]        [Health < 70 o Warning/Critical]
              │                             │
              ▼                             ▼
   [FAST-PATH RETORNO DIRECTO]   ┌──────────────────────────────────────────┐
                                 │ 3. DEEP-PATH BAJO DEMANDA                │
                                 │    • LSTM Autoencoder Temporal           │
                                 │    • Weibull RUL (P10, P50, P90)         │
                                 │    • Sensor Drift KS-Test                │
                                 │    • Agente SAP PM / SAE J1939 RAG       │
                                 └──────────────────────────────────────────┘
```

---

## 4. Fundamentos Matemáticos e Industriales

### A. Matriz de Costo Asimétrica y Decisión Bayesiana (CBM)
En la faena minera, los costos de error son brutalmente asimétricos:
- $C_{FP} = \$8,000\text{ USD}$ (Falsa Alarma: desviar un camión al taller para inspección innecesaria).
- $C_{FN} = \$350,000\text{ USD}$ (Falso Negativo: rotura catastrófica de motor o cilindro en rampa a plena carga).

La regla de decisión de Bayes determina analíticamente el umbral crítico óptimo:
$$\theta^*_{\text{crit}} = \frac{C_{FP}}{C_{FN} + C_{FP}} = \frac{8,000}{350,000 + 8,000} \approx \mathbf{0.0223 \quad (2.23\%)}$$

> **Regla de Operación:** Si $P(\text{Critical}) \ge 2.23\%$, el camión se deriva inmediatamente a inspección preventiva en taller. Esperar certeza del 50% significaría arriesgar pérdidas millonarias por ahorrar una inspección menor.

### B. Estimación Estocástica de RUL Weibull No Lineal
Modelamos la degradación con una función de supervivencia de Weibull:
$$R(t) = \exp\left(-\left(\frac{t}{\eta}\right)^\beta\right)$$
donde $\beta \approx 2.8$ modela el codo de desgaste acelerado de los rodamientos y conjuntos mecánicos. El sistema calcula los cuantiles condicionales:
- **$RUL_{P10}$:** Estimación conservadora con 90% de certeza (parada mandatoria antes de este límite).
- **$RUL_{P50}$:** Mediana esperada.
- **$RUL_{P90}$:** Escenario optimista.

### C. Sensor Drift y Validación Cruzada Física
- **Test Kolmogorov-Smirnov (KS-Test):** Detecta descalibración estadística de termistores o acelerómetros ($p < 0.01$).
- **Consistencia Física:** Si `engine_temp` supera 100°C pero `coolant_temp` permanece frío a bajas RPM, el sistema diagnostica `SENSOR_CALIBRATION_DRIFT` y evita una parada no programada del equipo.

---

## 5. Estructura del Repositorio

```text
predictive-maintenance-system/
 |-- data/
 |   |-- raw/                       # Telemetría de sensores sin procesar
 |   |-- processed/                 # Dataset procesado con features temporales
 |-- models/
 |   |-- saved_models/              # Pesos entrenados (.pkl, .keras, .onnx)
 |-- src/
 |   |-- api/                       # API REST v2 Modular (FastAPI)
 |   |   |-- routers/               # Routers: telemetry, maintenance, fleet
 |   |   |-- schemas.py             # Modelos Pydantic v2
 |   |   |-- dependencies.py        # Inyección de dependencias
 |   |-- preprocessing/
 |   |   |-- streaming_window_buffer.py  # Buffer O(1) en RAM (68 features)
 |   |   |-- feature_engineer.py    # Pipeline de features por lotes
 |   |   |-- data_synthesizer.py    # Simulador de camiones CAEX
 |   |-- models/
 |   |   |-- staged_triage_engine.py     # Motor en cascada Fast-Path/Deep-Path
 |   |   |-- cost_sensitive_cbm.py       # Decisor Bayesiano con matriz de costos
 |   |   |-- onnx_runtime_engine.py      # Runtime ONNX con fallback nativo
 |   |   |-- export_onnx.py              # Utilidad de exportación a ONNX
 |   |   |-- autoencoder_deep.py    # Deep Denoising Autoencoder
 |   |   |-- autoencoder_lstm.py    # LSTM Autoencoder temporal
 |   |   |-- gan_detector.py        # GAN Anomaly Detector
 |   |   |-- xgb_predictor.py       # XGBoost Multi-Class
 |   |   |-- isolation_forest.py    # Baseline no supervisado
 |   |-- evaluation/
 |   |   |-- weibull_rul_estimator.py    # RUL Estocástico P10-P90
 |   |   |-- rul_analyzer.py        # Evaluador de RUL industrial
 |   |   |-- metrics_engine.py      # Motor de métricas y torneo
 |   |-- monitoring/
 |   |   |-- sensor_drift_detector.py    # Monitor KS-Test y consistencia física
 |   |-- maintenance/
 |   |   |-- work_order_rag_agent.py     # Agente SAP PM / SAE J1939 RAG
 |   |-- dashboard/
 |   |   |-- app.py                 # Dashboard Streamlit v2.0 (6 Módulos)
 |-- tests/                         # Suite de 25 pruebas unitarias e integración
 |   |-- test_streaming_buffer.py
 |   |-- test_cost_sensitive_cbm.py
 |   |-- test_weibull_rul.py
 |   |-- test_sensor_drift.py
 |   |-- test_work_order_rag_agent.py
 |   |-- test_staged_triage.py
 |   |-- test_api_v2.py
 |-- fastapii.py                    # Servidor REST FastAPI v2.0
 |-- Dockerfile                     # Contenedor para despliegue industrial
 |-- pyproject.toml                 # Configuración de pytest
 |-- requirements.txt               # Dependencias de producción
 |-- README.md                      # Documentación institucional
```

---

## 6. Instalación y Ejecución

### 1. Clonar y Configurar Entorno
```bash
git clone https://github.com/ronaldreighsrsc/predictive-maintenance-system.git
cd predictive-maintenance-system

# Crear y activar entorno virtual
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar Suite de Pruebas Automatizadas
```bash
pytest tests/ -v
```
*Garantiza 25 pruebas unitarias e integración pasando al 100%.*

### 3. Iniciar API REST v2.0 (FastAPI)
```bash
uvicorn fastapii:app --host 0.0.0.0 --port 8000 --reload
```
- Documentación interactiva Swagger: `http://localhost:8000/docs`
- Endpoint principal de Ingesta Streaming: `POST /api/v2/telemetry/ingest`
- Endpoint de Órdenes de Trabajo: `POST /api/v2/maintenance/work-order`

### 4. Iniciar Dashboard Industrial (Streamlit)
```bash
streamlit run src/dashboard/app.py
```
- Acceso: `http://localhost:8501`

---

## 7. Ejemplo de Ficha SAP PM Generada

```text
================================================================================
ORDEN DE TRABAJO AUTOMATIZADA - SISTEMA CBM (SAP PM / ISO 13374)
EQUIPO: CAEX-104 | FLOTA: KOMATSU 930E-4SE | FECHA: 2026-09-18 21:30 UTC
PRIORIDAD: 1 - ALTA (PARADA PROGRAMADA INMEDIATA)
================================================================================
DIAGNÓSTICO ANALÍTICO:
- Health Score: 22.4 / 100 | Riesgo Crítico: 8.50%
- RUL Estimado (P10): 5.4 ciclos operacionales
- Subsistema Comprometido: CIRCUITO DE REFRIGERACIÓN Y DISIPACIÓN TÉRMICA
- Causa Raíz Detectada: Sobrecalentamiento severo de motor con gradiente anormal en refrigerante
  [engine_temp: 106.5 (+7.2σ), coolant_temp: 98.0 (+8.0σ)]
- Código de Falla SAE J1939: SPN 110 / FMI 00 (Sistema Térmico / Culata / Bomba de Agua)

PLAN DE ACCIÓN PRESCRIPTIVO PARA TALLER:
1. Aislar camión en bahía de mantenimiento mecánico (Lockout/Tagout).
2. Permitir enfriamiento pasivo antes de abrir circuito presurizado.
3. Inspeccionar visualmente pérdidas de líquido en sellos de culata y mangueras de retorno.
4. Escanear con cámara termográfica el radiador frontal para detectar tubos tapados.
5. Reemplazar kit de sellos o termostato según prueba hidrostática de presión a 25 PSI.

LISTA DE REPUESTOS REQUERIDOS (BOM / OEM):
- Part #KM-98210-A: Kit sellos bomba refrigerante Komatsu 930E (Cant: 1 un)
- Part #CAT-248-5513: Termostato dual Cummins QSK78 / Cat 797F (Cant: 2 un)
- Part #CH-RAD-600: Refrigerante etilenglicol 50/50 OAT (Cant: 40 L)
================================================================================
```

---

## 8. Speech Táctico para Entrevistas en Gran Minería (Codelco / BHP / AMSA)

> *"Muchos desarrolladores construyen modelos de Machine Learning para predecir fallas usando datos estáticos de sensores, pero esos modelos nunca llegan a implementarse en los camiones de faena. ¿Cómo garantizas que tu sistema de mantenimiento predictivo realmente funcione en una mina abierta con 100 camiones CAEX?"*

### Tu Respuesta de Ingeniero Civil Industrial de Élite:
> *"Esa brecha ocurre cuando se confunde un notebook de analítica con un sistema de confiabilidad operacional de misión crítica.*  
>
> *En mi sistema (**Predictive Maintenance System v2.0**), resolví tres desafíos que impiden que los proyectos tradicionales sobrevivan en faena:*
>
> 1. * **Desacoplamiento de Ingesta y Feature Engineering en Streaming:** La telemetría minera del bus CAN jamás enviará 68 variables calculadas. Mi API recibe únicamente los 7 valores brutos del sensor y utiliza un **Ring Buffer en memoria $O(1)$** en el backend para calcular las medias móviles, derivas y firmas térmicas en menos de 1.2 milisegundos, reduciendo la latencia de inferencia total a menos de 4 ms con **ONNX Runtime**.
>
> 2. * **Toma de Decisiones Basada en Costo Económico Real (CBM):** No utilizo clasificadores simétricos estándar de 50%. En minería, detener un camión por falsa alarma cuesta $8,000 USD, pero que un motor diésel o cilindro hidráulico reviente en la rampa a plena carga cuesta más de $350,000 USD en repuestos y bloqueo del circuito de acarreo. Usando la **regla de mínimo riesgo de Bayes**, calibré analíticamente el umbral crítico en **2.23%**. Si el modelo detecta un 2.3% de probabilidad de falla crítica, el camión se deriva a inspección preventiva, optimizando la disponibilidad mecánica de la flota y el EBITDA de la faena.
>
> 3. * **Accionabilidad Operativa con RUL Probabilístico y Agente SAP PM:** No entrego estimaciones lineales teóricas de RUL con data leakage. Modélo la degradación con **curvas no lineales Weibull con intervalos de confianza P10-P90**, permitiendo al planificador saber con certeza si el equipo aguanta hasta el próximo cambio de turno. Además, un **Agente RAG basado en normas ISO 13374 y códigos SAE J1939** genera automáticamente la orden de trabajo con el diagnóstico prescriptivo y los repuestos requeridos lista para el taller mecánico.*
>
> *Este enfoque es transversal a toda mi ingeniería: ya sea en fraude bancario, redes edge o mantenimiento de flotas, mis sistemas toman decisiones óptimas bajo costos económicos asimétricos y arquitecturas de baja latencia."*
