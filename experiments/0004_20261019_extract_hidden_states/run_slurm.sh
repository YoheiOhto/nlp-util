#!/bin/bash
# Variables only: runx sources this file on the login node and scripts/slurm_entry.sh
# sources it again inside the job. No #SBATCH lines, no commands with side effects.

# =====================================================
# Resources (plan.resources.per_job: 1 GPU, 16 CPU, 40000M, 01:00:00)
# Basis: Qwen3-8B fits one 96 GB GPU in bf16 (16 GB) and fp32 (32 GB); 16 prompts,
# one forward per prompt; the main cost is model load + GGUF sha256 + dump_hidden build.
# =====================================================
HEADER_TYPE="slurm_generic"
DEFAULT_GPU=1
DEFAULT_CPU=16
DEFAULT_MEM="40000M"
DEFAULT_TIME="01:00:00"

# =====================================================
# Runtime environment
# =====================================================
# llama.cpp image (contains /opt/llama.cpp headers/static libs, cmake); the login
# environment exports a different SIF_PATH without llama.cpp.
export SIF_PATH="${PROJECT_ROOT}/env/env.sif"
# Offline: the HF snapshot is read from the local cache only (plan.data.input_paths).
export HF_HUB_OFFLINE=1

# =====================================================
# Storage
# =====================================================
USE_LOCAL_SSD_INPUT=0

# =====================================================
# python path
# =====================================================
PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Multi run (runx --array): 7 tasks of plan.variants.tasks, one task per array element
# =====================================================
BASE_COMMAND="python ${PYTHON_PATH}"

GRID_ARGS=(
    "--task"
)

GRID_VALUES=(
    "hf-bf16 hf-fp32 vllm-bf16-hidden vllm-bf16-pooling llamacpp-bf16 llamacpp-q8_0 llamacpp-q4_k_m"
)
