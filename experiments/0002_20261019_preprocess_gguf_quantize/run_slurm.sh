#!/bin/bash

# =====================================================
# Resources (plan.md resources.per_job)
# No GPU, 16 CPUs, 20000M, 01:00:00 (30B Q8_0 took 1003 s before approval).
# runx builds the #SBATCH header, partition and USR1 margin from these values.
# llama-quantize links libcuda; slurm_entry.sh always runs apptainer --nv.
# Requires both 0001 variants to be completed (input SHA-256 is verified).
# =====================================================
HEADER_TYPE="slurm_generic"
DEFAULT_GPU=0
DEFAULT_CPU=16
DEFAULT_MEM="20000M"
DEFAULT_TIME="01:00:00"

# Approved runtime: env/env.sif -> llamacpp.sif (the login env exports another SIF_PATH).
export SIF_PATH="${PROJECT_ROOT}/env/env.sif"

# =====================================================
# Storage
# No data/ input (0001 outputs are read from outputs/ directly). OUTPUT_ROOT
# is not set, so experiment.py writes to outputs/ directly.
# =====================================================

USE_LOCAL_SSD_INPUT=0

# =====================================================
# python path
# =====================================================
PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Multi run (runx --array): one job per (model, type), 2 x 2 = 4 configs.
#   0: --model qwen3-8b      --type Q8_0    (qwen3-8b__q8_0)
#   1: --model qwen3-8b      --type Q4_K_M  (qwen3-8b__q4_k_m)
#   2: --model qwen3-30b-a3b --type Q8_0    (qwen3-30b-a3b__q8_0)
#   3: --model qwen3-30b-a3b --type Q4_K_M  (qwen3-30b-a3b__q4_k_m)
# =====================================================

BASE_COMMAND="python ${PYTHON_PATH}"

GRID_ARGS=(
    "--model"
    "--type"
)

GRID_VALUES=(
    "qwen3-8b qwen3-30b-a3b"
    "Q8_0 Q4_K_M"
)
