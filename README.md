# AQI Forecasting Workflow

A Pegasus workflow system for processing air quality sensor data from OpenAQ to detect anomalies and analyze pollutant trends. It also has LSTM-based AQI forecasting capabilities, enabling prediction of air quality for the next 24 hours based on historical patterns.

## Overview

The AQI forecasting workflow extends the base air quality analysis with machine learning capabilities:

- **Fetches historical data**: 90 days (configurable) of air quality measurements for training
- **Trains LSTM models**: Per-location PyTorch-based LSTM models for time-series prediction
- **Generates forecasts**: 24-hour (configurable) AQI predictions with confidence intervals
- **Visualizes results**: Plots showing historical data, forecasts, and confidence bounds

## Data Source

This workflow uses the [OpenAQ API v3](https://docs.openaq.org/) to fetch air quality measurements from monitoring stations worldwide.

### API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `https://api.openaq.org/v3/locations` | Search for monitoring locations |
| `https://api.openaq.org/v3/locations/{id}` | Get location details |
| `https://api.openaq.org/v3/locations/{id}/sensors` | Get sensors at a location |
| `https://api.openaq.org/v3/sensors/{id}/days` | Fetch daily aggregated measurements |

### Parameters Fetched

The `fetch_openaq_catalog.py` script retrieves the following pollutant measurements:

| Parameter | Description | OpenAQ ID | Typical Unit |
|-----------|-------------|-----------|--------------|
| `pm25` | Fine particulate matter (≤2.5 μm) | 2 | µg/m³ |
| `pm10` | Coarse particulate matter (≤10 μm) | 1 | µg/m³ |
| `o3` | Ozone | 5 | µg/m³ or ppm |
| `no2` | Nitrogen Dioxide | 3 | µg/m³ or ppm |
| `so2` | Sulfur Dioxide | 4 | µg/m³ or ppm |
| `co` | Carbon Monoxide | 7 | µg/m³ or ppm |

### Data Characteristics

- **Source**: Global network of air quality monitoring stations
- **Temporal Resolution**: Daily aggregates (min, max, median, mean, std)
- **Coverage**: Historical data from stations worldwide
- **API Key**: Required - register at https://explore.openaq.org/register

For more details, see the [OpenAQ API documentation](https://docs.openaq.org/).

### Customizing Parameters

To change or add parameters, modify the `fetch_openaq_catalog.py` script:

1. **Edit the default parameters list** in the `fetch_openaq_catalog()` function (around line 117):

```python
if parameters is None:
    parameters = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
```

2. **Update the parameter map** if adding new parameters (around line 120):

```python
parameter_map = {
    'pm25': 2,    # PM2.5
    'pm10': 1,    # PM10
    'o3': 5,      # Ozone
    'no2': 3,     # Nitrogen Dioxide
    'so2': 4,     # Sulfur Dioxide
    'co': 7       # Carbon Monoxide
    # Add new parameters here with their OpenAQ IDs
}
```

3. **Or use command-line arguments** to select specific parameters:

```bash
./fetch_openaq_catalog.py --location-ids 2178 --parameters pm25 pm10 o3 --start-date 2024-01-01
```

**Finding OpenAQ Parameter IDs**: Use the OpenAQ Explorer (https://explore.openaq.org/) or API to discover available parameters and their IDs for specific locations.

## Architecture

### Workflow Structure

```
Base Pipeline (runs in parallel):
  Data Fetch → Extract Timeseries → Analyze → Detect Anomalies → Merge

Forecast Pipeline:
  Data Fetch → Extract Timeseries ─────────┐
                                           ↓
          Fetch Historical Data → Prepare Features → Train LSTM → Generate Forecast → Visualize
```

### Components

| Component | Purpose | Memory | Container |
|-----------|---------|--------|-----------|
| `fetch_historical_data.py` | Fetch 90 days of historical AQI data | 2 GB | Forecast |
| `prepare_features.py` | Feature engineering for LSTM | 2 GB | Forecast |
| `train_forecast_model.py` | Train PyTorch LSTM model | 4 GB | Forecast |
| `generate_forecast.py` | Generate 24-hour predictions | 2 GB | Forecast |
| `visualize_forecast.py` | Create forecast visualizations | 2 GB | Forecast |

## Prerequisites

### Pegasus/HTCondor Cluster

Before running this workflow, you need a Pegasus/HTCondor cluster. This cluster can be deployed on any infrastructure that supports HTCondor and Pegasus WMS, including cloud providers (AWS, GCP, Azure), on-premises clusters, or research testbeds.

Below we provide an example of how to deploy the cluster on the [FABRIC testbed](https://fabric-testbed.net/):

#### Option 1: FABRIC Artifact (Recommended)

Download and run the pre-configured Pegasus-FABRIC notebook from the FABRIC Artifacts repository:

**Artifact URL**: https://artifacts.fabric-testbed.net/artifacts/53da4088-a175-4f0c-9e25-a4a371032a39

This artifact contains a complete setup for deploying a distributed Pegasus/HTCondor infrastructure across FABRIC sites.

#### Option 2: Jupyter Examples Repository

Use the Pegasus-FABRIC notebook from the official FABRIC Jupyter examples:

**GitHub**: https://github.com/fabric-testbed/jupyter-examples/blob/f7be0c75f22544c72d7b3e3fa42bbdfd9d8bb841/fabric_examples/complex_recipes/pegasus/pegasus-fabric.ipynb

#### Cluster Architecture

The notebook provisions:
- **Submit Node**: Central Manager running HTCondor scheduler and Pegasus WMS
- **Worker Nodes**: Distributed execution points across multiple FABRIC sites
- **FABNetv4 Networking**: Private L3 network connecting all nodes

#### Setup Steps

1. Log into the [FABRIC JupyterHub](https://jupyter.fabric-testbed.net/)
2. Upload/clone the Pegasus-FABRIC notebook
3. Configure your desired sites and node specifications
4. Run the notebook to provision the cluster
5. SSH to the submit node to run workflows

### Python Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Build Docker Container

```bash
cd Docker
docker build -f AirQuality_Forecast_Dockerfile -t kthare10/airquality-forecast:latest .
docker push kthare10/airquality-forecast:latest  # If using with Pegasus
```

### 2. Set OpenAQ API Key

The workflow requires an OpenAQ API key to fetch historical data:

```bash
# Set API key (required!)
export OPENAQ_API_KEY='your-api-key-here'

# Get your API key at: https://explore.openaq.org/register
```

### 3. Generate Workflow

```bash
# Basic usage
./workflow_generator.py \
    --location-ids 2178 \
    --start-date 2024-01-15 \
    --end-date 2024-01-16 \
    --output workflow_forecast.yml

# Advanced options
./workflow_generator.py \
    --location-ids 2178 1490 \
    --start-date 2024-01-15 \
    --historical-days 90 \
    --forecast-horizon 48 \
    --parameters pm25 pm10 o3 so2 \
    --execution-site condorpool \
    --output workflow_forecast.yml
```

### 4. Submit Workflow

```bash
pegasus-plan --submit -s condorpool -o local workflow_forecast.yml
```

### 5. Monitor Workflow

```bash
pegasus-status /path/to/submit/directory
pegasus-analyzer /path/to/submit/directory
```

## Command-Line Options

### workflow_generator.py

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--location-ids` | int+ | **Required** | OpenAQ location IDs to analyze |
| `--start-date` | YYYY-MM-DD | **Required** | Analysis start date |
| `--end-date` | YYYY-MM-DD | start_date + 1 day | Analysis end date |
| `--historical-days` | int | 90 | Days of historical data for training |
| `--forecast-horizon` | int | 24 | Hours to forecast ahead |
| `--parameters` | str+ | All 6 | Pollutants: pm25, pm10, o3, no2, so2, co |
| `--execution-site` | str | condorpool | HTCondor execution site |
| `-o, --output` | str | workflow_forecast.yml | Output YAML file |

## LSTM Model Architecture

### Model Configuration

```python
Input: (batch, 168 timesteps, ~24 features)
  ↓
LSTM Layer 1 (hidden_size=128, dropout=0.2)
  ↓
LSTM Layer 2 (hidden_size=128, dropout=0.2)
  ↓
Fully Connected Layer (hidden_size → 24 predictions)
  ↓
Output: (batch, 24 hourly AQI predictions)
```

### Features

The model uses 24 features per timestep:

**Raw Features (7)**:
- AQI value
- PM2.5, PM10, O3, NO2, SO2, CO concentrations

**Temporal Features (6, cyclic encoded)**:
- Hour of day (sin/cos)
- Day of week (sin/cos)
- Month (sin/cos)

**Statistical Features (12)**:
- Rolling mean (6h, 12h, 24h windows)
- Rolling std (6h, 12h, 24h windows)

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Lookback | 168 hours (7 days) | Historical window for input |
| Horizon | 24 hours | Prediction length |
| Batch Size | 32 | Training batch size |
| Epochs | 100 | Maximum epochs (with early stopping) |
| Patience | 10 | Early stopping patience |
| Learning Rate | 0.001 | Adam optimizer initial LR |
| Loss Function | MSE | Mean Squared Error |
| Validation Split | 20% | Train/validation split ratio |

## Output Files

### Directory Structure

```
output/
├── timeseries/
│   └── <location>/
│       └── <location>_timeseries.json        # Recent AQI measurements
├── historical/
│   └── <location>/
│       ├── <location>_historical.csv          # 90 days of training data
│       └── <location>_coverage.json           # Data completeness info
├── features/
│   └── <location>/
│       ├── <location>_train.npz               # Feature matrices (X, y)
│       └── <location>_train_scaler.json       # Normalization parameters
├── models/
│   └── <location>/
│       ├── <location>_lstm_checkpoint.pt      # Trained model weights
│       └── <location>_training_info.json      # Training metadata
├── forecasts/
│   └── <location>/
│       ├── <location>_forecast.json           # 24-hour predictions
│       ├── <location>_forecast.png            # Visualization
│       └── <location>_forecast_summary.json   # Statistics
├── analysis/                                   # Base workflow outputs
│   └── <location>/
│       ├── <location>_analysis.png
│       └── <location>_statistics.json
└── anomalies/                                  # Base workflow outputs
    └── <location>/
        └── <location>_anomalies.json
```

### Forecast JSON Format

```json
{
  "location": "London N. Kensington",
  "forecast_generated": "2024-01-15T10:00:00Z",
  "forecast_start": "2024-01-16T00:00:00Z",
  "forecast_horizon_hours": 24,
  "model_info": {
    "architecture": "LSTM",
    "input_size": 24,
    "hidden_size": 128,
    "num_layers": 2
  },
  "predictions": [
    {
      "datetime": "2024-01-16T00:00:00Z",
      "predicted_aqi": 45.2,
      "confidence_interval_lower": 35.6,
      "confidence_interval_upper": 54.8,
      "predicted_category": "Good"
    },
    // ... 23 more hours
  ]
}
```

## Visualization

The forecast visualization includes:

1. **Historical AQI** (last 7 days): Solid blue line with markers
2. **Forecast** (next 24 hours): Dashed orange line with markers
3. **Confidence Intervals**: Shaded orange region (95% CI)
4. **AQI Categories**: Background color bands (Good=green, Moderate=yellow, etc.)
5. **Separation Line**: Vertical line marking forecast start
6. **Legend**: Model info and horizon

## Troubleshooting

### Common Issues

**Issue**: `Insufficient data coverage` warning
- **Cause**: Location has <70% data completeness
- **Solution**: Increase `--historical-days` or choose location with better coverage

**Issue**: `Training failed with NaN loss`
- **Cause**: Insufficient data or poor feature quality
- **Solution**: Check data coverage, try different location, or adjust normalization

**Issue**: `Model checkpoint not found`
- **Cause**: Training job failed or didn't complete
- **Solution**: Check Pegasus logs, verify training completed successfully

**Issue**: `Container pull failed`
- **Cause**: Docker image not available or Singularity issues
- **Solution**: Build and push Docker image, or check Singularity configuration

### Debugging Tips

```bash
# Check workflow status
pegasus-status /path/to/submit/dir

# View job logs
pegasus-analyzer /path/to/submit/dir

# Check specific job output
cat /path/to/submit/dir/*/*.out
cat /path/to/submit/dir/*/*.err

# Verify historical data coverage
cat output/historical/<location>/<location>_coverage.json

# Check training progress
cat output/models/<location>/<location>_training_info.json
```

## References

- OpenAQ API: https://docs.openaq.org/
- PyTorch Documentation: https://pytorch.org/docs/
- Pegasus WMS: https://pegasus.isi.edu/
- EPA AQI Standards: https://www.airnow.gov/aqi/aqi-basics/

## Support

For issues or questions:
- GitHub Issues: https://github.com/your-repo/issues
- FABRIC Support: https://learn.fabric-testbed.net/

## Authors
Komal Thareja (kthare10@renci.org)

P.S: Built with the assistance of [Claude](https://claude.ai), Anthropic's AI assistant.