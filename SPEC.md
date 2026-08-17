# SPEC.md — Air Quality Forecasting Workflow

**Version**: 1.0
**Status**: Implemented (test harness and roadmap items are planned)
**Audience**: Developers who want to rebuild, port, or extend this workflow from scratch.

This document is a self-contained build specification. Given only this file, a
developer should be able to reimplement the workflow and verify it against the
acceptance criteria in §9.

---

## 1. Purpose and Scope

A [Pegasus WMS](https://pegasus.isi.edu/) workflow that processes air quality
sensor data to:

1. **Analyze** pollutant trends per monitoring location (plots + statistics).
2. **Detect anomalies** in AQI time series (z-score based).
3. **Forecast** AQI for the next 24 hours (configurable) using a per-location
   PyTorch LSTM trained on 90 days (configurable) of historical data.

### Data sources

| Source | Mode | Pipelines run |
|--------|------|---------------|
| **OpenAQ API v3** (default) | Live HTTP fetch at workflow *generation* time + at *execution* time for historical data | Base + Forecast |
| **SAGE Continuum** | Live query via `sage_data_client`, or offline JSONL file | Base only (forecast is force-skipped — SAGE lacks the OpenAQ historical API needed for training) |

### Out of scope

- Real-time/streaming ingestion (batch only).
- Multi-pollutant forecasting (the LSTM predicts a single AQI target).
- Model serving — forecasts are produced as static JSON/PNG artifacts.

---

## 2. System Requirements

### Submit node

- Python ≥ 3.8 with packages from `requirements.txt`:
  `pandas>=1.5.0`, `numpy>=1.23.0`, `matplotlib>=3.5.0`, `scipy>=1.9.0`,
  `requests>=2.28.0`, `sage-data-client>=0.7.0`, `pytz>=2022.1`, `torch`,
  `scikit-learn>=1.0.0`, `tqdm>=4.62.0`, `pegasus-wms>=5.0.0`
- Pegasus WMS ≥ 5.0 and HTCondor installed and configured.
- `OPENAQ_API_KEY` environment variable set (register free at
  <https://explore.openaq.org/register>). Required only for the OpenAQ source.

### Execution nodes

- HTCondor `vanilla` universe, Singularity/Apptainer available.
- Network access to `api.openaq.org` (the `fetch_historical` job calls the API
  at execution time).
- ≥ 4 GB RAM per slot (largest job memory profile).

### Container

A single image serves all jobs: a locally built Apptainer image referenced as
`file://…/Apptainer/AirQuality_Forecast_Container.sif` with
`image_site="local"`. Pegasus stages the `.sif` like any other input file, so
there is no registry pull. Build spec
(`Apptainer/AirQuality_Forecast_Container.def`):

- Bootstrap: `docker`, From: `python:3.8.12-slim` (Apptainer pulls and converts
  the OCI base itself — no Docker installation required to build)
- System: `git wget curl build-essential`. **`curl` and `wget` are mandatory**:
  PegasusLite downloads its worker package inside the container before the job
  script runs.
- Python: `pandas numpy matplotlib scipy requests pytz sage-data-client torch
  scikit-learn tqdm`
- `/app/output` created in `%post` (Apptainer has no `WORKDIR`; jobs `cd` into
  their own working directory at run time); `%runscript` execs `/bin/bash`

The `.sif` path is overridable with `--container-sif`. A `.sif` carries **one
architecture** — there is no multi-arch manifest — so it MUST be built on a host
matching the worker nodes; Apptainer cannot build on macOS at all. See
`../APPTAINER.md`. `Docker/AirQuality_Forecast_Dockerfile` is retained as a
fallback.

---

## 3. Architecture

### 3.1 Pipeline overview

Two pipelines share a common fan-in point (`extract_timeseries`) and run in
parallel per location:

```
mkdir → extract_timeseries ──┬──→ analyze_pollutants
                             ├──→ detect_anomalies ──→ merge   (only if >1 location)
                             └──→ prepare_features ←── fetch_historical_data
                                       └──→ train_model → generate_forecast → visualize_forecast
```

Key dependency rule: `prepare_features` depends on **both**
`extract_timeseries` and `fetch_historical_data`, which run in parallel.
`mkdir` is the root job for each location and runs on the `local` site.

### 3.2 Job specification

All jobs except `mkdir` run inside the container on the execution site.
Per-location job IDs follow the pattern `<verb>_<location>` where `location`
is the sanitized location name (see §3.4).

| Transformation | Script | Memory | Inputs (LFN) | Outputs (LFN) | Staged out |
|---|---|---|---|---|---|
| `mkdir` | `/bin/mkdir -p` (local, not stageable) | — | — | output dirs | — |
| `extract_timeseries` | `bin/extract_aqi_timeseries.py` | 2 GB | `openaq_catalog.csv` | `timeseries/<loc>/<loc>_timeseries.json` | no |
| `analyze_pollutants` | `bin/analyze_pollutants.py` | 2 GB | timeseries JSON | `analysis/<loc>/<loc>_analysis.png`, `analysis/<loc>/<loc>_statistics.json` | yes |
| `detect_anomalies` | `bin/detect_anomalies.py` | 1 GB | timeseries JSON | `anomalies/<loc>/<loc>_anomalies.json` | yes |
| `merge` | `bin/merge.py` | 1 GB | all anomaly JSONs | `merged_anomalies.json` | yes |
| `fetch_historical` | `bin/fetch_historical_data.py` | 2 GB | — (calls OpenAQ API) | `historical/<loc>/<loc>_historical.csv` | no |
| `prepare_features` | `bin/prepare_features.py` | 2 GB | timeseries JSON + historical CSV | `features/<loc>/<loc>_train.npz`, `features/<loc>/<loc>_train_scaler.json` | no |
| `train_model` | `bin/train_forecast_model.py` | 4 GB | features `.npz` + scaler | `models/<loc>/<loc>_lstm_checkpoint.pt`, `models/<loc>/<loc>_training_info.json` | yes |
| `generate_forecast` | `bin/generate_forecast.py` | 2 GB | checkpoint + timeseries + scaler | `forecasts/<loc>/<loc>_forecast.json` | yes |
| `visualize_forecast` | `bin/visualize_forecast.py` | 2 GB | timeseries + forecast JSON | `forecasts/<loc>/<loc>_forecast.png`, `forecasts/<loc>/<loc>_forecast_summary.json` | yes |

Job CLI contracts (as invoked by the generator):

```
extract_timeseries  -i openaq_catalog.csv -o timeseries/<loc>
analyze_pollutants  -i timeseries/<loc>/<loc>_timeseries.json -o analysis/<loc>
detect_anomalies    -i timeseries/<loc>/<loc>_timeseries.json -o anomalies/<loc>/<loc>_anomalies.json -t 3.0
merge               -i <anomaly1.json> <anomaly2.json> ... -o merged_anomalies.json
fetch_historical    --location-id <id> --days <N> --end-date YYYY-MM-DD --output historical/<loc>/<loc>_historical.csv
prepare_features    --timeseries <json> --historical <csv> --output features/<loc>/<loc>_train.npz --lookback 168 --horizon <H>
train_model         --features <npz> --output models/<loc> --location-name <loc> --epochs 100 --batch-size 32 --patience 10
generate_forecast   --model <ckpt> --timeseries <json> --scaler <json> --output forecasts/<loc>/<loc>_forecast.json --location-name "<display>" --lookback 168
visualize_forecast  --timeseries <json> --forecast <json> --output forecasts/<loc>/<loc>_forecast.png --lookback-days 7
```

Secrets: `OPENAQ_API_KEY` is injected into `fetch_historical` jobs only, via
`Job.add_env(OPENAQ_API_KEY=os.environ.get('OPENAQ_API_KEY', ''))` — i.e. the
key is captured at generation time and embedded in the submit files.

### 3.3 Workflow generator

`workflow_generator.py` (entry point, class `AirQualityForecastWorkflow`) is
responsible for:

1. **Pegasus Properties** — `pegasus.transfer.threads = 16`.
2. **Site Catalog** — `local` site with `SHARED_SCRATCH` (`./scratch`) and
   `LOCAL_STORAGE` (`./output`) directories with `file://` file servers;
   execution site (default `condorpool`) with Condor profile
   `universe=vanilla` and Pegasus profile `style=condor`. Skippable with
   `--skip-sites-catalog` (for pre-configured clusters).
3. **Transformation Catalog** — registers the container plus the 10
   transformations above; all `bin/*.py` are `is_stageable=True`.
4. **Replica Catalog** — registers `openaq_catalog.csv`. This file is
   produced **at generation time**:
   - OpenAQ: call `fetch_openaq_catalog()` (imported from
     `fetch_openaq_catalog.py`) with location IDs, dates, parameters.
   - SAGE: read JSONL (or `sage_data_client.query`), filter by
     `--sage-vsn` / `--sage-plugin` / `--sage-names`, map measurement names to
     parameters (`env.air_quality.conc`/`env.pm25` → `pm25`; `env.pm10` →
     `pm10`; everything else dropped), normalize to the same CSV schema.
5. **DAG construction** — per location, with `infer_dependencies=True` plus
   explicit dependencies for `mkdir`, `fetch_hist`, and `prepare`.
   Jobs are grouped with Pegasus `label` profiles: `<loc>` for the base
   pipeline, `<loc>_forecast` for the forecast pipeline (enables label-based
   clustering).

Generator CLI:

| Option | Type / default | Notes |
|---|---|---|
| `--location-ids` | int+, required for OpenAQ | OpenAQ location IDs |
| `--start-date` | `YYYY-MM-DD`, required | analysis window start |
| `--end-date` | default `start + 1 day` | |
| `--parameters` | subset of `pm25 pm10 o3 no2 so2 co`, default all | |
| `--historical-days` | 90 | LSTM training window |
| `--forecast-horizon` | 24 | hours predicted |
| `--data-source` | `openaq` \| `sage` | |
| `--sage-input` / `--sage-vsn` / `--sage-plugin` / `--sage-names` | optional | SAGE filters; `--sage-input` required if `sage_data_client` missing |
| `--skip-forecast` | flag | forced `true` for SAGE |
| `-e, --execution-site-name` | `condorpool` | |
| `-s, --skip-sites-catalog` | flag | |
| `-o, --output` | `workflow_forecast.yml` | |

Validation rules: OpenAQ requires `--location-ids`; SAGE without
`sage_data_client` requires `--sage-input`; SAGE + forecast emits a warning
and sets `skip_forecast=True`; empty catalog → exit 1.

### 3.4 Naming conventions

- **Location sanitization**: location display names become path components by
  replacing spaces, hyphens, and slashes with underscores
  (`"London N. Kensington"` → `London_N._Kensington`).
- All per-location artifacts follow `<category>/<loc>/<loc>_<suffix>`.

---

## 4. Data Contracts

### 4.1 Catalog CSV (`openaq_catalog.csv`)

Long-format CSV, one row per (location, parameter, timestamp):
`location, location_id, parameter, value, unit, datetime` (SAGE rows
additionally carry `timestamp` epoch seconds and `hour_bucket`). Parameter
values use OpenAQ canonical names: `pm25, pm10, o3, no2, so2, co`
(OpenAQ parameter IDs 2, 1, 5, 3, 4, 7).

### 4.2 Timeseries JSON

Per-location AQI time series extracted from the catalog; consumed by analyze,
anomaly, prepare, forecast, and visualize steps.

### 4.3 Historical CSV

Wide-format daily/hourly pollutant table covering `--days` before
`--end-date`, plus a sibling `<loc>_coverage.json` reporting completeness.
Coverage < 70% should produce a warning (see Troubleshooting in README).

### 4.4 Feature matrices (`.npz` + scaler JSON)

- `X`: `(num_samples, 168, 24)` float32 — 168-hour lookback windows.
- `y`: `(num_samples, horizon)` — future AQI targets.
- Scaler JSON stores normalization parameters so inference can apply the
  identical transform.

**24 features per timestep**:
- 7 raw: AQI, pm25, pm10, o3, no2, so2, co
- 6 cyclic temporal: hour-of-day, day-of-week, month — each sin+cos encoded
- 12 rolling stats: mean and std over 6h/12h/24h windows (currently 6 values
  documented per stat family; keep mean+std × 3 windows × 2 = 12)

### 4.5 Model checkpoint and training info

PyTorch checkpoint (`.pt`) + JSON with training metadata (epochs run, losses,
early-stopping info).

**Model architecture** (must match exactly for checkpoint compatibility):

```
Input (batch, 168, 24)
 → LSTM layer 1 (hidden=128, dropout=0.2)
 → LSTM layer 2 (hidden=128, dropout=0.2)
 → Fully connected (128 → horizon)
Output (batch, horizon)   # hourly AQI predictions
```

**Training config**: batch 32, max 100 epochs, early stopping patience 10,
Adam lr=0.001, MSE loss, 20% validation split.

### 4.6 Forecast JSON

```json
{
  "location": "<display name>",
  "forecast_generated": "ISO-8601",
  "forecast_start": "ISO-8601",
  "forecast_horizon_hours": 24,
  "model_info": {"architecture": "LSTM", "input_size": 24, "hidden_size": 128, "num_layers": 2},
  "predictions": [
    {"datetime": "...", "predicted_aqi": 45.2,
     "confidence_interval_lower": 35.6, "confidence_interval_upper": 54.8,
     "predicted_category": "Good"}
  ]
}
```

AQI categories follow EPA AirNow bands (Good, Moderate, Unhealthy for
Sensitive Groups, Unhealthy, Very Unhealthy, Hazardous).

### 4.7 Anomaly JSON

Z-score detection with default threshold **3.0** (job arg `-t 3.0`);
`merge.py` concatenates per-location files into `merged_anomalies.json`.

### 4.8 Output tree (LOCAL_STORAGE = `./output`)

```
output/
├── timeseries/<loc>/   # intermediate (not staged out)
├── analysis/<loc>/     # PNG + statistics JSON           (staged out)
├── anomalies/<loc>/    # anomaly JSON                    (staged out)
├── historical/<loc>/   # 90-day CSV + coverage JSON      (intermediate)
├── features/<loc>/     # .npz + scaler JSON              (intermediate)
├── models/<loc>/       # checkpoint .pt + training JSON  (staged out)
└── forecasts/<loc>/    # forecast JSON + PNG + summary   (staged out)
```

---

## 5. Build and Run Procedure

```bash
# 0. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAQ_API_KEY='your-api-key'

# 1. Container (only when changing dependencies). No registry push — Pegasus
#    stages the .sif. Must run on a Linux host matching the workers.
apptainer build Apptainer/AirQuality_Forecast_Container.sif \
    Apptainer/AirQuality_Forecast_Container.def

# 2. Find locations
./fetch_openaq_catalog.py --search --city "Los Angeles"

# 3. Generate the DAG
./workflow_generator.py --location-ids 2178 --start-date 2024-01-15 \
    --output workflow_forecast.yml

# 4. Plan + submit
pegasus-plan --submit -s condorpool -o local workflow_forecast.yml

# 5. Monitor / debug
pegasus-status  <submit-dir>
pegasus-analyzer <submit-dir>
```

### Deployment targets

| Target | How |
|---|---|
| ACCESS | `Access-Airquality-workflow.ipynb` notebook (Pegasus/HTCondor pre-configured) |
| FABRIC (manual) | Pegasus-FABRIC artifact / `pegasus-fabric.ipynb` to provision submit + workers over FABNetv4, then run as above |
| FABRIC (Kiso) | `experiment.yml` / `kiso-airquality-experiment.yml`: `kiso check && kiso up && kiso run && kiso down`. Secrets in `secrets/fabric_rc.py` and `secrets/openaq_api_key` |

---

## 6. Test Harness (specification — to be implemented)

There is currently **no automated test suite**. A reimplementation MUST add
the following harness under `tests/`, runnable with
`pytest` (and wired into CI per §7).

### 6.1 Layout

```
tests/
├── conftest.py                  # fixtures: tmp output tree, sample catalog, frozen clock
├── fixtures/
│   ├── openaq_catalog_small.csv     # 2 locations × 6 params × ~3 days
│   ├── sage_sample.jsonl            # mixed VSNs/plugins/names incl. non-matching rows
│   ├── historical_90d.csv           # synthetic sinusoidal AQI with known anomalies
│   └── timeseries_london.json
├── unit/
│   ├── test_generator_dag.py
│   ├── test_sage_loader.py
│   ├── test_extract_timeseries.py
│   ├── test_detect_anomalies.py
│   ├── test_prepare_features.py
│   ├── test_train_model.py
│   ├── test_generate_forecast.py
│   └── test_merge.py
├── integration/
│   ├── test_pipeline_local.py       # run bin/ scripts end-to-end on fixtures, no Pegasus
│   └── test_pegasus_plan.py         # pegasus-plan dry-run (skipped if pegasus absent)
└── e2e/
    └── test_condorpool_smoke.sh     # full submit on a real pool (manual/nightly)
```

### 6.2 Required test cases

**Generator / DAG (no network — mock `fetch_openaq_catalog`)**
1. Single location → DAG contains exactly 9 jobs (mkdir, extract, analyze,
   anomaly, fetch_hist, prepare, train, forecast, viz) and **no** merge job.
2. Two locations → 18 jobs + 1 `merge_all_anomalies` job whose inputs are both
   anomaly LFNs.
3. `--skip-forecast` → only mkdir/extract/analyze/anomaly per location.
4. Dependency assertions: `prepare_<loc>` has parents `extract_<loc>` **and**
   `fetch_hist_<loc>`; `mkdir_<loc>` is root.
5. Location sanitization: `"London N. Kensington"`, `"A-B/C"` → underscores;
   no `/`, space, or `-` in job IDs or LFN path components.
6. `OPENAQ_API_KEY` present in `fetch_hist` job env; absent from all others.
7. Validation errors: OpenAQ without `--location-ids` exits non-zero; SAGE
   without input and without `sage_data_client` exits non-zero; SAGE forces
   `skip_forecast=True` with a warning.
8. Generated YAML round-trips through `Pegasus.api` loader / passes
   `pegasus-plan` dry-run.

**SAGE loader**
9. JSONL filtering by VSN, plugin, and names each drop non-matching rows.
10. Name mapping: `env.air_quality.conc` → `pm25`, `env.pm10` → `pm10`,
    unknown names dropped; malformed JSON lines skipped without error.
11. Output CSV has required columns and UTC-parsed datetimes; rows with
    unparseable dates dropped; empty result → exit code 1.

**Processing scripts (each invoked via its CLI on fixtures)**
12. `extract_aqi_timeseries.py`: produces one JSON per location with
    monotonically increasing timestamps and AQI values computed from the
    fixture.
13. `detect_anomalies.py`: synthetic series with injected spikes at known
    indices → exactly those points flagged at `-t 3.0`; threshold `-t 100`
    flags none.
14. `prepare_features.py`: output `X.shape == (N, 168, 24)`,
    `y.shape == (N, horizon)`; scaler JSON inverts correctly
    (`inverse(transform(x)) ≈ x`); insufficient history (< lookback+horizon)
    fails with a clear message.
15. `train_forecast_model.py`: train 2 epochs on a tiny fixture → checkpoint
    loads, `training_info.json` records losses; loss decreases on a learnable
    synthetic signal; NaN-input data → non-zero exit with diagnostic.
16. `generate_forecast.py`: with a fixed-seed checkpoint, output JSON matches
    schema §4.6 (validate with `jsonschema`), has exactly `horizon`
    predictions, `lower ≤ predicted ≤ upper`, valid EPA category strings.
17. `merge.py`: merging N files preserves all anomalies and location keys.
18. `visualize_forecast.py`: produces a non-empty PNG and summary JSON
    (use `matplotlib` Agg backend).

**Integration**
19. `test_pipeline_local.py`: chain steps 12→14→15→16→18 in a tmp dir using
    only fixtures (no network, `OPENAQ_API_KEY` unset) and assert the §4.8
    tree exists with all staged-out artifacts.
20. `test_pegasus_plan.py`: generate YAML + `pegasus-plan --dir … --dax`
    (no submit); skip with `pytest.mark.skipif` when `pegasus-plan` is not on
    PATH.

**E2E smoke (manual / nightly, real credentials)**
21. One known-good location (e.g. 2178), 1-day window, reduced
    `--historical-days 30`; submit to a condorpool; poll `pegasus-status`
    until done; assert exit 0 and forecast JSON exists.

### 6.3 Conventions

- All unit/integration tests MUST run offline: HTTP via `requests` is mocked
  (`responses` or `unittest.mock`); `sage_data_client` is monkeypatched.
- Seed `torch`, `numpy`, and `random` in `conftest.py` for reproducibility.
- Target runtime: unit+integration < 5 min on a laptop CPU.
- Coverage gate: ≥ 80% on `workflow_generator.py` and `bin/`.

### 6.4 Tooling

```bash
pip install pytest pytest-cov responses jsonschema
pytest tests/unit tests/integration --cov=bin --cov=. --cov-report=term-missing
```

---

## 7. CI/CD (planned)

GitHub Actions workflow `.github/workflows/ci.yml`:

| Job | Trigger | Steps |
|---|---|---|
| `lint` | PR, push | `ruff check`, `ruff format --check` |
| `test` | PR, push | Python 3.8 + 3.11 matrix; install `requirements.txt` + test deps; run §6 unit+integration |
| `container` | tag `v*` or `Apptainer/**` change | `apptainer build Apptainer/AirQuality_Forecast_Container.sif Apptainer/AirQuality_Forecast_Container.def`, smoke-test (`apptainer exec … python -c "import torch, pandas, sklearn"`), publish the `.sif` as a release artifact |
| `e2e-smoke` | nightly cron, manual dispatch | §6.2 case 21 against a self-hosted condorpool runner (secrets: `OPENAQ_API_KEY`) |

The `container` job MUST run on an `x86_64` Linux runner: a `.sif` has no
multi-arch manifest, so an image built elsewhere will not exec on the worker
nodes. Published `.sif` filenames MUST carry the git SHA, and the generator
already accepts `--container-sif` so runs are reproducible against a pinned
image.

---

## 8. Roadmap

### M1 — Hardening (near-term)
- [ ] Implement the test harness of §6 and CI of §7.
- [ ] Pin container image by digest/tag; add `--container-tag` generator flag.
- [ ] Pass `OPENAQ_API_KEY` via Pegasus credentials/condor secrets instead of
      embedding it in submit files via `add_env`.
- [ ] Schema-validate all JSON artifacts (`jsonschema`) at job boundaries.
- [ ] Upgrade container base from Python 3.8 (EOL) to 3.11+ and align with
      `requirements.txt` version pins.
- [ ] Retry/backoff + rate-limit handling for OpenAQ API calls.

### M2 — Capability
- [ ] SAGE forecast support: train on SAGE history (drop the OpenAQ-only
      assumption) when enough edge data exists.
- [ ] Hyperparameter flags surfaced through the generator
      (`--hidden-size`, `--lookback`, `--epochs`, …).
- [ ] Model evaluation job: held-out MAE/RMSE per location, written to
      `models/<loc>/<loc>_evaluation.json`; fail workflow if error exceeds a
      configurable bound.
- [ ] Checkpoint reuse: warm-start training from a prior run's checkpoint to
      cut training time on repeated runs.
- [ ] Optional GPU profile for `train_model` (Condor `request_gpus`).

### M3 — Scale and edge-to-cloud
- [ ] Edge tier variant aligned with `airquality-workflow-dpu`: preprocessing
      (extract/prepare) routed to BlueField-3 DPU ARM nodes, training/inference
      to cloud nodes via HTCondor ClassAds (see `EDGE_TO_CLOUD_GUIDE.md`).
- [ ] Multi-location batch training (single job, shared encoder) to reduce
      per-location overhead at 50+ locations.
- [ ] Pegasus ensemble manager integration for recurring daily forecasts.
- [ ] Data provenance: register staged-out artifacts in the replica catalog
      (`register_replica=True`) and capture OpenAQ fetch metadata.

### M4 — Research / stretch
- [ ] Alternative models (Transformer, N-BEATS) behind a `--model-arch` flag
      with a common checkpoint contract.
- [ ] Probabilistic forecasts (quantile loss) replacing the current CI
      heuristic.
- [ ] Cross-location transfer learning for stations with sparse history.

---

## 9. Acceptance Criteria

A reimplementation is conformant when:

1. `./workflow_generator.py --location-ids 2178 --start-date 2024-01-15`
   produces a YAML that `pegasus-plan` accepts, containing the job set, memory
   profiles, and dependencies of §3.2 exactly.
2. A full run on a condorpool with a valid `OPENAQ_API_KEY` completes with all
   jobs successful and the staged-out tree of §4.8 populated.
3. The forecast JSON validates against §4.6 with `horizon` predictions and
   monotonically increasing datetimes.
4. SAGE mode (`--data-source sage` with the JSONL fixture) runs the base
   pipeline only and produces analysis + anomaly artifacts.
5. The §6 test suite passes offline with ≥ 80% coverage of generator + bin.

---

## 10. Reference Files

| File | Role |
|---|---|
| `workflow_generator.py` | DAG generator (this spec §3.3) |
| `fetch_openaq_catalog.py` | OpenAQ fetch/search; importable module |
| `bin/*.py` | Job implementations (§3.2) |
| `Apptainer/AirQuality_Forecast_Container.def` | Container build (§2); `Docker/AirQuality_Forecast_Dockerfile` retained as fallback |
| `requirements.txt` | Submit-node Python deps |
| `experiment.yml`, `kiso-airquality-experiment.yml` | Kiso/FABRIC deployment |
| `Access-Airquality-workflow.ipynb` | ACCESS walkthrough |
| `README.md` | User-facing docs; `CLAUDE.md` — agent guidance |
