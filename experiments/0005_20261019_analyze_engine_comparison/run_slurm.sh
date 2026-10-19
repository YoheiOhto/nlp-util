#!/bin/bash
# Variables only: runx sources this file on the login node and scripts/slurm_entry.sh
# sources it again inside the job. No #SBATCH lines, no commands with side effects.

# =====================================================
# Resources (plan.resources.per_job: 0 GPU, 16 CPU, 40000M, 01:00:00)
# Basis: CPU-only float64 metrics over ~7 x 16 prompts x 39 layers of (T, 4096)
# arrays, loaded one (prompt, layer) at a time; parsing of 30 x 5 bench JSON files.
# =====================================================
HEADER_TYPE="slurm_generic"
DEFAULT_GPU=0
DEFAULT_CPU=16
DEFAULT_MEM="40000M"
DEFAULT_TIME="01:00:00"

# =====================================================
# Runtime environment
# =====================================================
export SIF_PATH="${PROJECT_ROOT}/env/env.sif"
export MPLBACKEND=Agg

# =====================================================
# Storage
# =====================================================
USE_LOCAL_SSD_INPUT=0

# =====================================================
# python path
# =====================================================
PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Single run (runx 0005)
# =====================================================
RUN_COMMAND="python ${PYTHON_PATH}"
