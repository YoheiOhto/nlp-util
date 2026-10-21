#!/bin/bash
# Usage:
#   bash tools/review.sh [exp_name_or_id]

set -euo pipefail

EXP_NAME=${1:-}
PROJECT_ROOT=$(git rev-parse --show-toplevel)

if [ ! -d "${PROJECT_ROOT}/experiments" ]; then
    echo "❌ Error: 'experiments' directory not found."
    exit 1
fi

if [ -n "$EXP_NAME" ]; then
    # If the input is only numbers, pad to 4 digits
    if [[ "$EXP_NAME" =~ ^[0-9]+$ ]]; then
        SEARCH_STR=$(printf "%04d" "$EXP_NAME")
    else
        SEARCH_STR="$EXP_NAME"
    fi

    # find the experiment directory (resolving prefixes)
    TARGET_DIR=$(find "${PROJECT_ROOT}/experiments" -maxdepth 1 -type d -name "*${SEARCH_STR}*" | head -n 1)
    if [ -z "$TARGET_DIR" ]; then
        echo "❌ Error: Experiment matching '${SEARCH_STR}' not found."
        exit 1
    fi
    
    EXP_BASE=$(basename "$TARGET_DIR")
    OUT_FILE="${TARGET_DIR}/review.md"
    TEMPLATE_FILE="${PROJECT_ROOT}/templates/review.md"
    
    if [ ! -f "$TEMPLATE_FILE" ]; then
        echo "❌ Error: template not found at ${TEMPLATE_FILE}"
        exit 1
    fi

    echo "Creating review for experiment: ${EXP_BASE}"
    sed -e "s/<実験名>/${EXP_BASE}/" \
        -e "s/NNNN_YYYYMMDD_<name>/${EXP_BASE}/" \
        "$TEMPLATE_FILE" > "$OUT_FILE"
    echo "✅ Created ${OUT_FILE}"
else
    echo "❌ Error: experiment name or id is required. Usage: make review exp=<id_or_name>" >&2
    exit 1
fi