#!/bin/bash
 
# =====================================================
# Resources (qsub/PBS)
# ⚠️ 注意: DEFAULT_TIME/DEFAULT_MEM は既存実験の値をコピーして使う。自分で概算しない。
# ⚠️ 注意: シェル上での for/while ループによる実行は禁止です。
# =====================================================
 
HEADER_TYPE="miyabi_gpu"
DEFAULT_GPU=1
DEFAULT_CPU=20
DEFAULT_MEM="110g"
DEFAULT_TIME="04:00:00"
 
# =====================================================
# Storage
# =====================================================
 
USE_LOCAL_SSD_INPUT=0
 
 
# =====================================================
# python path
# =====================================================
PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
 
 
# =====================================================
# Single run (runx)
# =====================================================
 
# RUN_COMMAND="
# python ${PYTHON_PATH} \
#     --config config.yml
# "
 
# =====================================================
# Multi run (runx --array or runx --seq)
# =====================================================
 
# BASE_COMMAND="python ${PYTHON_PATH}"
 
# GRID_ARGS=(
#     "--model"
#     "--dataset"
# )
 
# GRID_VALUES=(
#     "google/gemma-4-31b-it meta-llama/Llama-3-8b-it Qwen/Qwen3.6-27B mistralai/Ministral-3-14B-Instruct-2512 google/medgemma-27b-text-it BioMistral/BioMistral-7B openai/gpt-oss-20b"
#     "BC5CDR BIORED"
# )
