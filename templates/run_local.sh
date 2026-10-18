#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="__PROJECT_ROOT__"
export EXP_NAME="__EXP_NAME__"
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0
RUN_MODE="single"
PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_COMMAND="python3 ${PYTHON_PATH} --config config.yml"
source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
