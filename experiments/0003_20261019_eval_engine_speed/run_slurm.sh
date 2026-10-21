#!/bin/bash
# Settings only: runx sources this file on the login node to read them, and
# scripts/slurm_entry.sh sources it again inside the job. No side effects.

# =====================================================
# Resources (plan.md resources.per_job)
# - 1 GPU; 16 CPUs: llama-server --threads 16 (plan) plus the CPU-only
#   vllm bench client.
# - 40000M host RAM: plan value; no measured history yet.
# - 2 h: one server load + 5 workloads (864 requests x 256 output tokens)
#   + GGUF SHA-256 + llama-bench for llama.cpp tasks.
# runx builds the #SBATCH header, resolves the partition from DEFAULT_TIME
# and sets the USR1 time-limit warning margin.
# =====================================================
HEADER_TYPE="slurm_generic"
DEFAULT_GPU=1
DEFAULT_CPU=16
DEFAULT_MEM="40000M"
DEFAULT_TIME="02:00:00"

# =====================================================
# Container
# The login environment exports a different SIF_PATH that lacks llama.cpp.
# Set it only inside the job so sourcing this file on the login node does not
# change the caller's shell.
# =====================================================
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export SIF_PATH="${PROJECT_ROOT}/env/env.sif"
fi

# =====================================================
# Storage
# Inputs (HF snapshots, GGUFs) are read in place; outputs go directly to
# outputs/ (OUTPUT_ROOT is not set by slurm_entry.sh).
# =====================================================
USE_LOCAL_SSD_INPUT=0

# =====================================================
# python path
# =====================================================
PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Multi run (runx --array): 1 job = 1 plan task (model x engine_format x repeat)
# 2 x 5 x 3 = 30 tasks; expansion order equals plan.md variants.tasks.
# =====================================================
BASE_COMMAND="python ${PYTHON_PATH}"

GRID_ARGS=(
    "--model"
    "--engine_format"
    "--repeat"
)

GRID_VALUES=(
    "qwen3-8b qwen3-30b-a3b"
    "vllm-bf16 vllm-fp8 llamacpp-bf16 llamacpp-q8_0 llamacpp-q4_k_m"
    "1 2 3"
)
