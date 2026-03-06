#!/bin/bash
# Wrapper script invoked by Kiso to generate and submit the air quality
# Pegasus workflow on the HTCondor submit node.
#
# All arguments are forwarded to workflow_generator.py (e.g. --location-ids,
# --start-date). Kiso passes the args defined in experiment.yml.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Load OpenAQ API key from the uploaded secrets file
OPENAQ_KEY_FILE="${SCRIPT_DIR}/secrets/openaq_api_key"
if [[ -f "${OPENAQ_KEY_FILE}" ]]; then
    export OPENAQ_API_KEY="$(cat "${OPENAQ_KEY_FILE}" | tr -d '[:space:]')"
    echo "OpenAQ API key loaded."
else
    echo "WARNING: ${OPENAQ_KEY_FILE} not found. OPENAQ_API_KEY must already be set." >&2
fi

WORKFLOW_FILE="workflow_forecast.yml"

# Step 1: Generate the Pegasus workflow DAG
echo "=== Generating workflow ==="
python3 workflow_generator.py \
    --output "${WORKFLOW_FILE}" \
    "$@"

# Step 2: Plan and submit to the local HTCondor pool
echo "=== Submitting workflow ==="
pegasus-plan \
    --submit \
    --sites condorpool \
    --output-sites local \
    "${WORKFLOW_FILE}"
