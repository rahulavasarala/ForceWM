#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../environment.yml"
OPENSAI_CONDA_ENV="${OPENSAI_CONDA_ENV:-opensai-dev}"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: Missing Conda environment file at $ENV_FILE"
    exit 1
fi

if command -v mamba >/dev/null 2>&1; then
    CONDA_SOLVER="mamba"
elif command -v conda >/dev/null 2>&1; then
    CONDA_SOLVER="conda"
else
    echo "ERROR: Neither conda nor mamba was found in PATH."
    exit 1
fi

if conda env list | awk 'NR > 2 {print $1}' | grep -qx "$OPENSAI_CONDA_ENV"; then
    echo "Updating Conda environment '$OPENSAI_CONDA_ENV' with $CONDA_SOLVER."
    "$CONDA_SOLVER" env update --yes --name "$OPENSAI_CONDA_ENV" --file "$ENV_FILE" --prune
else
    echo "Creating Conda environment '$OPENSAI_CONDA_ENV' with $CONDA_SOLVER."
    "$CONDA_SOLVER" env create --yes --name "$OPENSAI_CONDA_ENV" --file "$ENV_FILE"
fi

echo "Conda environment '$OPENSAI_CONDA_ENV' is ready."
echo "Next step: bash ${SCRIPT_DIR}/install_core_libraries.sh"
