# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **Pegasus WMS workflow** for processing air quality sensor data. It runs two parallel pipelines:
- **Base pipeline**: fetch → extract timeseries → analyze pollutants → detect anomalies → merge
- **Forecast pipeline**: fetch historical → prepare features → train LSTM → generate forecast → visualize

Data sources: **SAGE Continuum** edge sensors (default) and **OpenAQ API v3**. SAGE runs the base pipeline only (no forecast).

Every generator argument has a default, so the workflow can be launched from the Pegasus Studio GUI with nothing filled in. List arguments also accept the GUI's single comma-separated token (`--sage-vsn W045,W123`) in addition to the shell's space-separated form.

## Setup

```bash
pip install -r requirements.txt
export OPENAQ_API_KEY='your-api-key'  # Only for --data-source openaq
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
# Generate workflow YAML with all defaults (SAGE node W045, last full UTC day)
./workflow_generator.py

# Generate workflow YAML (SAGE, explicit)
./workflow_generator.py --data-source sage --sage-vsn W045 \
    --sage-plugin registry.sagecontinuum.org/seanshahkarami/air-quality:0.3.0 \
    --sage-names env.air_quality.conc --start-date 2026-01-14 --output workflow_forecast.yml

# Generate workflow YAML (OpenAQ, by location ID)
./workflow_generator.py --data-source openaq --location-ids 2178 \
    --start-date 2024-01-15 --output workflow_forecast.yml

# Generate workflow YAML (OpenAQ, by named region — IDs resolved live)
./workflow_generator.py --data-source openaq --openaq-region los-angeles \
    --start-date 2024-01-15 --output workflow_forecast.yml

# Find OpenAQ location IDs manually
./fetch_openaq_catalog.py --search --city "Los Angeles"

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
| `bin/fetch_sage_data.py` | **Job** — queries SAGE for one VSN and writes a catalog CSV (keeps `sage_data_client` inside the container) |
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
SAGE (default) — one branch per --sage-vsn, forecast pipeline skipped:
mkdir → fetch_sage → extract_timeseries ──┬──→ analyze_pollutants
                                           └──→ detect_anomalies → merge (if multiple VSNs)

OpenAQ — catalog already in the replica catalog, so no fetch job:
mkdir → extract_timeseries ──┬──→ analyze_pollutants
                              ├──→ detect_anomalies → merge (if multiple locations)
                              └──→ prepare_features ←── fetch_historical_data
                                        └──→ train_model → generate_forecast → visualize_forecast
```

`prepare_features` depends on **both** `extract_timeseries` and `fetch_historical_data` (they run in parallel).

### Pegasus Catalogs (created by `workflow_generator.py`)

- **Sites Catalog**: defines `local` (scratch + storage dirs) and execution site (default: `condorpool`)
- **Transformation Catalog**: registers all `bin/*.py` scripts with their containers and memory profiles (1–4 GB)
- **Replica Catalog**: for OpenAQ, registers `openaq_catalog.csv` (fetched at generation time). For SAGE, registers only an optional `--sage-input` JSONL dump — the catalog itself is produced at run time by `fetch_sage`

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

- The `OPENAQ_API_KEY` env var is passed directly into `fetch_hist` jobs via `.add_env()` — it must be set before running `workflow_generator.py` with `--data-source openaq`. The generator fails loudly at generation time if it is missing, rather than letting jobs fail on a worker node.
- SAGE needs no credentials, and `sage_data_client` only has to exist inside the container: `bin/fetch_sage_data.py` runs as the `fetch_sage` job. For SAGE, locations come from `--sage-vsn` rather than from the data, because the catalog does not exist until the job runs.
- `fetch_sage` always writes its declared output CSV even when the query fails or returns nothing, then exits non-zero. Skipping the write would make HTCondor HOLD the job on a stage-out error and hang the DAG.
- SAGE data automatically forces `--skip-forecast` (SAGE lacks the OpenAQ historical API needed for training).
- Location names are sanitized (spaces/hyphens/slashes → underscores) for use as file path components.
- The notebook `Access-Airquality-workflow.ipynb` is the recommended way to run on ACCESS/FABRIC clusters with Pegasus pre-configured.
