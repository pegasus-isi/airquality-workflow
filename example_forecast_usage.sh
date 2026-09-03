#!/bin/bash

# Example usage script for AQI Forecasting Workflow
# This demonstrates how to use the forecasting extension

set -e  # Exit on error

echo "=========================================="
echo "AQI Forecasting Workflow - Example Usage"
echo "=========================================="
echo ""

# Configuration
LOCATION_ID=2178  # London N. Kensington
START_DATE="2024-01-15"
END_DATE="2024-01-16"
HISTORICAL_DAYS=30  # Using 30 days for faster testing (90 recommended for production)
FORECAST_HORIZON=24

echo "Configuration:"
echo "  Location ID: $LOCATION_ID"
echo "  Start Date: $START_DATE"
echo "  End Date: $END_DATE"
echo "  Historical Days: $HISTORICAL_DAYS"
echo "  Forecast Horizon: $FORECAST_HORIZON hours"
echo ""

# Check for OpenAQ API key
if [ -z "$OPENAQ_API_KEY" ]; then
    echo "ERROR: OPENAQ_API_KEY environment variable not set"
    echo "Please register at https://explore.openaq.org/register and set:"
    echo "  export OPENAQ_API_KEY='your-api-key'"
    exit 1
fi

echo "✓ OpenAQ API key found"
echo ""

# ========================================
# Option 1: Standard Forecast Workflow
# ========================================
echo "=========================================="
echo "OPTION 1: Standard Forecast Workflow"
echo "=========================================="
echo ""

echo "Generating standard forecast workflow..."
./workflow_generator.py \
    --data-source openaq \
    --location-ids $LOCATION_ID \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --historical-days $HISTORICAL_DAYS \
    --forecast-horizon $FORECAST_HORIZON \
    --output workflow_forecast.yml

echo ""
echo "✓ Workflow generated: workflow_forecast.yml"
echo ""
echo "To submit this workflow:"
echo "  pegasus-plan --submit -s condorpool -o local workflow_forecast.yml"
echo ""

# ========================================
# Option 2: Edge-to-Cloud DPU Workflow
# ========================================
echo "=========================================="
echo "OPTION 2: Edge-to-Cloud DPU Workflow"
echo "=========================================="
echo ""

echo "Generating edge-to-cloud forecast workflow..."
./workflow_generator_dpu.py \
    --location-ids $LOCATION_ID \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --historical-days $HISTORICAL_DAYS \
    --forecast-horizon $FORECAST_HORIZON \
    --edge-site edgepool \
    --cloud-site cloudpool \
    --output workflow_forecast_dpu.yml

echo ""
echo "✓ Workflow generated: workflow_forecast_dpu.yml"
echo ""
echo "To submit this workflow:"
echo "  pegasus-plan --submit -s edgepool -s cloudpool -o local workflow_forecast_dpu.yml"
echo ""

# ========================================
# Multi-location Example
# ========================================
echo "=========================================="
echo "OPTION 3: Multi-location Forecast"
echo "=========================================="
echo ""

echo "Generating multi-location forecast workflow..."
./workflow_generator.py \
    --data-source openaq \
    --location-ids 2178 8118 \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --historical-days $HISTORICAL_DAYS \
    --forecast-horizon $FORECAST_HORIZON \
    --parameters pm25 pm10 o3 \
    --output workflow_forecast_multi.yml

echo ""
echo "✓ Workflow generated: workflow_forecast_multi.yml"
echo ""

# ========================================
# Docker Build Instructions
# ========================================
echo "=========================================="
echo "Docker Container Build Instructions"
echo "=========================================="
echo ""
echo "Before submitting workflows, build and push Docker containers:"
echo ""
echo "# Standard forecast container (with PyTorch). No registry push needed —"
echo "# Pegasus stages the .sif like any other input file. Apptainer cannot"
echo "# build on macOS and a .sif is single-architecture; see APPTAINER.md."
echo "  apptainer build Apptainer/AirQuality_Forecast_Container.sif \\"
echo "      Apptainer/AirQuality_Forecast_Container.def"
echo ""
echo "# DPU edge container (lightweight, no PyTorch) lives in the separate"
echo "# airquality-workflow-dpu repo; no .def in this workflow yet."
echo ""

# ========================================
# Output Structure
# ========================================
echo "=========================================="
echo "Expected Output Structure"
echo "=========================================="
echo ""
echo "After workflow completion, check:"
echo ""
echo "output/"
echo "├── timeseries/<location>/"
echo "│   └── <location>_timeseries.json"
echo "├── historical/<location>/"
echo "│   ├── <location>_historical.csv"
echo "│   └── <location>_coverage.json"
echo "├── features/<location>/"
echo "│   ├── <location>_train.npz"
echo "│   └── <location>_train_scaler.json"
echo "├── models/<location>/"
echo "│   ├── <location>_lstm_checkpoint.pt"
echo "│   └── <location>_training_info.json"
echo "├── forecasts/<location>/"
echo "│   ├── <location>_forecast.json          # 24-hour predictions"
echo "│   ├── <location>_forecast.png           # Visualization"
echo "│   └── <location>_forecast_summary.json  # Statistics"
echo "├── analysis/<location>/                  # Base workflow outputs"
echo "└── anomalies/<location>/                 # Base workflow outputs"
echo ""

# ========================================
# Monitoring Commands
# ========================================
echo "=========================================="
echo "Workflow Monitoring Commands"
echo "=========================================="
echo ""
echo "After submitting workflow:"
echo ""
echo "# Check status"
echo "  pegasus-status /path/to/submit/dir"
echo ""
echo "# View logs"
echo "  pegasus-analyzer /path/to/submit/dir"
echo ""
echo "# Monitor in real-time"
echo "  watch -n 10 'pegasus-status /path/to/submit/dir'"
echo ""

# ========================================
# Testing Individual Scripts
# ========================================
echo "=========================================="
echo "Testing Individual Scripts (Optional)"
echo "=========================================="
echo ""
echo "You can test individual forecast scripts manually:"
echo ""
echo "# 1. Fetch historical data"
echo "  ./bin/fetch_historical_data.py \\"
echo "      --location-id $LOCATION_ID \\"
echo "      --days $HISTORICAL_DAYS \\"
echo "      --output test_output/historical.csv"
echo ""
echo "# 2. Prepare features"
echo "  ./bin/prepare_features.py \\"
echo "      --timeseries test_output/timeseries.json \\"
echo "      --historical test_output/historical.csv \\"
echo "      --output test_output/train.npz"
echo ""
echo "# 3. Train model"
echo "  ./bin/train_forecast_model.py \\"
echo "      --features test_output/train.npz \\"
echo "      --output test_output/models/ \\"
echo "      --location-name test_location \\"
echo "      --epochs 10"  # Reduced for testing
echo ""
echo "# 4. Generate forecast"
echo "  ./bin/generate_forecast.py \\"
echo "      --model test_output/models/test_location_lstm_checkpoint.pt \\"
echo "      --timeseries test_output/timeseries.json \\"
echo "      --scaler test_output/train_scaler.json \\"
echo "      --output test_output/forecast.json"
echo ""
echo "# 5. Visualize"
echo "  ./bin/visualize_forecast.py \\"
echo "      --timeseries test_output/timeseries.json \\"
echo "      --forecast test_output/forecast.json \\"
echo "      --output test_output/forecast.png"
echo ""

echo "=========================================="
echo "✓ Example script complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Set OPENAQ_API_KEY environment variable"
echo "  2. Build Docker containers (see instructions above)"
echo "  3. Choose a workflow option (standard or DPU)"
echo "  4. Submit workflow to Pegasus"
echo "  5. Monitor execution"
echo "  6. Review forecast outputs in output/forecasts/"
echo ""
echo "For detailed documentation, see: README_FORECAST.md"
echo ""
