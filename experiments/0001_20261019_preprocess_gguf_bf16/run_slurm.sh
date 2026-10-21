#!/bin/bash

# =====================================================
# Resources (plan.md resources.per_job)
# No GPU, 16 CPUs, 40000M, 02:00:00 (30B conversion took 1926 s before approval).
# runx builds the #SBATCH header, partition and USR1 margin from these values.
# llama.cpp binaries link libcuda; slurm_entry.sh always runs apptainer --nv.
# =====================================================
HEADER_TYPE="slurm_generic"
DEFAULT_GPU=0
DEFAULT_CPU=16
DEFAULT_MEM="40000M"
DEFAULT_TIME="02:00:00"

# Approved runtime: env/env.sif -> llamacpp.sif (the login env exports another SIF_PATH).
export SIF_PATH="${PROJECT_ROOT}/env/env.sif"

# =====================================================
# Storage
# No data/ input (HF cache is read directly). OUTPUT_ROOT is not set, so
# experiment.py writes to outputs/ directly.
# =====================================================

USE_LOCAL_SSD_INPUT=0

# =====================================================
# python path
# =====================================================
PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Multi run (runx --array): one job per model (variant id == --model value).
#   0: --model qwen3-8b
#   1: --model qwen3-30b-a3b
# =====================================================

BASE_COMMAND="python ${PYTHON_PATH}"

GRID_ARGS=(
    "--model"
)

GRID_VALUES=(
    "qwen3-8b qwen3-30b-a3b"
)
