# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **Pegasus WMS workflow** for processing air quality sensor data. It runs two parallel pipelines:
- **Base pipeline**: fetch → extract timeseries → analyze pollutants → detect anomalies → merge
- **Forecast pipeline**: fetch historical → prepare features → train LSTM → generate forecast → visualize

Data sources: **OpenAQ API v3** (default) and **SAGE Continuum** edge sensors. SAGE runs the base pipeline only (no forecast).

## Setup

```bash
pip install -r requirements.txt
export OPENAQ_API_KEY='your-api-key'  # Required for OpenAQ data source
```

The container used for execution is `kthare10/airquality-forecast:latest` (Python 3.8 + PyTorch + scikit-learn).

```bash
# Build the Apptainer image if needed (from the workflow root). No registry push
# — Pegasus stages the .sif like any other input file. Apptainer cannot build on
# macOS and a .sif is single-architecture; see APPTAINER.md.
apptainer build Apptainer/AirQuality_Forecast_Container.sif \
    Apptainer/AirQuality_Forecast_Container.def
```

## Workflow Commands

```bash
# Find OpenAQ location IDs
./fetch_openaq_catalog.py --search --city "Los Angeles"

# Generate workflow YAML (OpenAQ)
./workflow_generator.py --location-ids 2178 --start-date 2024-01-15 --output workflow_forecast.yml

# Generate workflow YAML (SAGE, live query)
./workflow_generator.py --data-source sage --sage-vsn W045 \
    --sage-plugin registry.sagecontinuum.org/seanshahkarami/air-quality:0.3.0 \
    --sage-names env.air_quality.conc --start-date 2026-01-14 --output workflow_forecast.yml

# Submit to HTCondor
pegasus-plan --submit -s condorpool -o local workflow_forecast.yml

# Monitor / debug
pegasus-status /path/to/submit/dir
pegasus-analyzer /path/to/submit/dir
```

## Architecture

### Key Files

| File | Purpose |
|------|---------|
| `workflow_generator.py` | Entry point — generates the Pegasus DAG YAML via `AirQualityForecastWorkflow` class |
| `fetch_openaq_catalog.py` | Fetches/searches OpenAQ API v3; also importable as a module by the workflow generator |
| `bin/extract_aqi_timeseries.py` | Extracts per-location timeseries JSON from catalog CSV |
| `bin/analyze_pollutants.py` | Generates analysis PNG + statistics JSON |
| `bin/detect_anomalies.py` | Z-score anomaly detection (default threshold: 3.0) |
| `bin/merge.py` | Merges per-location anomaly files into one |
| `bin/fetch_historical_data.py` | Fetches 90 days of history from OpenAQ for LSTM training |
| `bin/prepare_features.py` | Feature engineering: 168-step lookback, cyclic temporal encoding, rolling statistics |
| `bin/train_forecast_model.py` | Trains 2-layer PyTorch LSTM (hidden=128, epochs=100, early stopping patience=10) |
| `bin/generate_forecast.py` | Runs trained model to produce 24-hour predictions with confidence intervals |
| `bin/visualize_forecast.py` | Renders forecast PNG with historical overlay and AQI category bands |
| `Apptainer/AirQuality_Forecast_Container.def` | Container definition for all execution jobs (legacy `Docker/AirQuality_Forecast_Dockerfile` kept as a fallback) |

### Workflow DAG Dependency

```
mkdir → extract_timeseries ──┬──→ analyze_pollutants
                              ├──→ detect_anomalies → merge (if multiple locations)
                              └──→ prepare_features ←── fetch_historical_data
                                        └──→ train_model → generate_forecast → visualize_forecast
```

`prepare_features` depends on **both** `extract_timeseries` and `fetch_historical_data` (they run in parallel).

### Pegasus Catalogs (created by `workflow_generator.py`)

- **Sites Catalog**: defines `local` (scratch + storage dirs) and execution site (default: `condorpool`)
- **Transformation Catalog**: registers all `bin/*.py` scripts with their containers and memory profiles (1–4 GB)
- **Replica Catalog**: registers `openaq_catalog.csv` (fetched at generation time, not at execution time)

### Output Directory Structure

```
output/
├── timeseries/<location>/     # JSON timeseries (intermediate, not staged out)
├── analysis/<location>/       # PNG + statistics JSON (staged out)
├── anomalies/<location>/      # Anomaly JSON (staged out)
├── historical/<location>/     # 90-day CSV (intermediate)
├── features/<location>/       # .npz feature matrices + scaler JSON (intermediate)
├── models/<location>/         # LSTM checkpoint .pt + training info JSON (staged out)
└── forecasts/<location>/      # Forecast JSON + PNG + summary JSON (staged out)
```

### LSTM Model

Input: `(batch, 168 timesteps, 24 features)` → 2×LSTM(128) → FC → `(batch, 24)` hourly AQI predictions.
Features: 7 raw pollutant values, 6 cyclic temporal encodings (hour/day/month sin+cos), 12 rolling stats (mean+std at 6h/12h/24h).

## Important Notes

- The `OPENAQ_API_KEY` env var is passed directly into `fetch_hist` jobs via `.add_env()` — it must be set before running `workflow_generator.py`.
- SAGE data automatically forces `--skip-forecast` (SAGE lacks the OpenAQ historical API needed for training).
- Location names are sanitized (spaces/hyphens/slashes → underscores) for use as file path components.
- The notebook `Access-Airquality-workflow.ipynb` is the recommended way to run on ACCESS/FABRIC clusters with Pegasus pre-configured.
