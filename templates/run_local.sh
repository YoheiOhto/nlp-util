#!/bin/bash
 
# =====================================================
# Resources (Local/Non-job)
# ⚠️ 注意: ローカル環境での実行は、必ず runx (ローカル疑似ジョブ経由) で行ってください。
# ⚠️ 注意: 直接 python コマンドで重いスクリプトを実行するのは禁止です。
# =====================================================
HEADER_TYPE="local"
DEFAULT_GPU=1
DEFAULT_CPU=4
DEFAULT_MEM="16g"
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
