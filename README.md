# Predictive Maintenance System

This repository implements an end-to-end **Predictive Maintenance System** using Anomaly Detection techniques from the NVIDIA Applications of AI for Anomaly Detection certification. It uses Deep Autoencoders for equipment health scoring, XGBoost for multi-class state classification (Normal/Warning/Critical), and Isolation Forest as an unsupervised baseline.

The project demonstrates the full ML lifecycle applied to industrial equipment monitoring: synthetic sensor data generation (mining truck simulation), temporal feature engineering, model training with anti-leakage cross-validation, comparative evaluation with industrial metrics, and interactive visualization.

---

## Project Architecture

The code is structured following modularity and single-responsibility principles (SOLID), dividing the workflow into three sequential blocks:

```text
predictive-maintenance-system/
 |-- data/
 |   |-- raw/                    # Synthetic sensor readings (generated or external datasets)
 |   |-- processed/              # Engineered features with rolling stats and degradation indicators
 |-- src/
 |   |-- preprocessing/          # Block 1: Data Synthesis, ETL, Feature Engineering
 |   |   |-- data_synthesizer.py # Mining truck sensor simulator with degradation patterns
 |   |   |-- data_loader.py      # Data loading and validation
 |   |   |-- feature_engineer.py # Temporal feature engineering (rolling, delta, EMA, cross-sensor)
 |   |-- models/                 # Block 2: Prediction Engines
 |   |   |-- autoencoder.py      # Deep Autoencoder for Health Score (0-100)
 |   |   |-- xgb_predictor.py    # XGBoost Multi-Class (Normal/Warning/Critical)
 |   |   |-- isolation_forest.py # Baseline: Isolation Forest (unsupervised)
 |   |-- evaluation/             # Block 3: Model Tournament & Industrial Analysis
 |   |   |-- metrics_engine.py   # Multi-class metrics + industrial KPIs
 |   |   |-- rul_analyzer.py     # Remaining Useful Life (RUL) analysis per equipment
 |   |   |-- results/            # Output artifacts (.npy, .csv)
 |   |-- dashboard/              # Interactive Streamlit Dashboard
 |   |   |-- app.py              # Multi-page application (4 pages)
 |   |   |-- pages_utils.py      # Plotly visualization utilities
 |   |-- main_preprocessing.py   # Block 1 Orchestrator
 |   |-- main_training.py        # Block 2 Orchestrator (Model Tournament)
 |   |-- main_evaluation.py      # Block 3 Orchestrator (Industrial Results)
 |-- .github/
 |   |-- workflows/
 |   |   |-- ci.yml              # Continuous Integration (lint + smoke test)
 |-- requirements.txt
 |-- README.md
```

---

## Installation and Setup

1. **Clone the repository** and open the terminal in the project root.
2. **Create and activate a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Execution Workflow (Pipeline)

To reproduce the experiments, execute the following scripts in order:

### Block 1: Data Preprocessing
Generates synthetic sensor data for 50 mining trucks with 7 sensors each. Simulates progressive degradation patterns before failures and applies temporal feature engineering: rolling statistics (mean, std), delta features, EMA, and cross-sensor interactions.
```bash
python src/main_preprocessing.py
```
*(Generates: `data/processed/sensors_engineered.csv`)*

### Block 2: Model Training
Trains three predictive models with equipment-level train/test split (80/20). The Autoencoder generates a **Health Score (0-100)** per reading. XGBoost classifies equipment into **Normal/Warning/Critical** states using Purged K-Fold CV.
```bash
python src/main_training.py
```
*(Generates: `.npy` prediction files, health scores, and `feature_importances.csv`)*

### Block 3: Evaluation & RUL Analysis
Runs the model tournament with multi-class metrics (Accuracy, Precision, Recall, Macro F1), industrial KPIs (False Alarm Rate, Critical Detection Rate, Estimated Cost), and **Remaining Useful Life (RUL)** analysis per equipment.
```bash
python src/main_evaluation.py
```
*(Generates: Tournament table in console, equipment RUL summary, and `maintenance_tournament.csv`)*

### Interactive Dashboard (Visualization)
Launches a **Streamlit** web application with four interactive pages: Fleet Overview (sensor timelines), Health Monitoring (Health Score vs RUL), Model Tournament (comparative metrics), and Sensor Explorer (correlation analysis).
```bash
streamlit run src/dashboard/app.py
```
*(Opens a local web server at `http://localhost:8501` with interactive Plotly charts).*

---

## Implemented Models

1. **Deep Denoising Autoencoder:** Symmetric architecture generating a Health Score (0-100) based on reconstruction error. Trained only on normal operation data. Injects Gaussian Noise to be robust against sensor vibration. Health Score < 60 → Warning, < 30 → Critical.
2. **LSTM Autoencoder:** Advanced Deep Learning architecture using Long Short-Term Memory cells to capture temporal relationships.
3. **XGBoost Multi-Class:** Gradient Boosting classifier for 3-state prediction (Normal/Warning/Critical) with Purged & Embargoed K-Fold CV and Macro F1 optimization.
4. **Isolation Forest (Baseline):** Unsupervised ensemble method for anomaly detection without labels.

---

## Synthetic Sensor Simulation

The `data_synthesizer.py` module simulates **mining truck sensors** with:
- **7 Sensors:** Engine temp, oil pressure, vibration, RPM, fuel consumption, coolant temp, hydraulic pressure.
- **Progressive Degradation:** Sensors drift gradually before failure (linear trend + increasing noise).
- **3 Equipment States:** Normal → Warning (30% before failure) → Critical (10% before failure).
- **RUL Labels:** Remaining Useful Life calculated at each cycle.

> **Real-World Compatibility:** The pipeline is compatible with the [NASA C-MAPSS Turbofan Engine Degradation](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) dataset. To use real data, simply replace the CSV in `data/raw/` and update the column mappings in `data_loader.py`.

---

## Certifications & Methodology

- **NVIDIA:** Applications of AI for Anomaly Detection
- **Anti-Leakage:** Purged K-Fold Cross-Validation with Embargo (López de Prado, 2018)
- **Industrial Metrics:** False Alarm Rate, Critical Detection Rate, Downtime Cost Analysis
- **RUL Analysis:** Remaining Useful Life prediction with alert lead time per equipment

---

## 🚀 V2.0: Big Data & GPU Scalability (NVIDIA RAPIDS)

While the current main branch is designed for CPU execution (Pandas/Scikit-Learn) to ensure maximum compatibility for anyone cloning the repo, this architecture is fully prepared to scale to massive industrial datasets (e.g., millions of sensor readings per day).

To achieve extreme performance without CPU-to-GPU memory bottlenecks, the ETL and training pipelines can be migrated to **NVIDIA RAPIDS**:

1. **cuDF (GPU Pandas):** Replaces traditional Pandas for Feature Engineering (rolling windows, EWMA). Data is loaded directly into GPU VRAM.
2. **Apache Arrow:** Eliminates serialization overhead. Data remains in the GPU memory while moving from the cuDF preprocessing step directly into the model.
3. **XGBoost (GPU Accelerated):** Uses 	ree_method='gpu_hist' to train on the GPU directly from the cuDF data structure, dropping training times from hours to seconds.

*Note: An experimental implementation of this GPU pipeline is available in the eature/rapids-gpu-scaling branch.*
