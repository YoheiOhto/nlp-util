#!/usr/bin/env bash
#PBS -N __EXP_NAME__
#PBS -o __PROJECT_ROOT__/logs/__EXP_NAME__/pbs.out
#PBS -e __PROJECT_ROOT__/logs/__EXP_NAME__/pbs.err
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=20:mem=110gb:ngpus=1
#PBS -V
set -euo pipefail
export PROJECT_ROOT="__PROJECT_ROOT__"
export EXP_NAME="__EXP_NAME__"
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0
RUN_MODE="single"
PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_COMMAND="python3 ${PYTHON_PATH} --config config.yml"
source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
