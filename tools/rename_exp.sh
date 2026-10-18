#!/bin/bash
# Usage:
#
# Rename latest experiment:
#   bash tools/rename_exp.sh new_name
#
# Rename experiment 0004:
#   bash tools/rename_exp.sh 4 better_baseline
#
# Rename by explicit experiment name:
#   bash tools/rename_exp.sh 0042_20260509_baseline roberta_large
#

set -euo pipefail

INPUT=${1:-}
NEW_LABEL=${2:-}

if [ -z "$INPUT" ] || [ -z "$NEW_LABEL" ]; then
    echo "❌ Usage: bash tools/rename_exp.sh <exp_id_or_name> <new_name>"
    exit 1
fi

# =========================================================
# Resolve project root
# =========================================================

PROJECT_ROOT=$(git rev-parse --show-toplevel)

EXP_ROOT="${PROJECT_ROOT}/experiments"
OUT_ROOT="${PROJECT_ROOT}/outputs"
LOG_ROOT="${PROJECT_ROOT}/logs"

# =========================================================
# Helper: resolve experiment dir from ID or name
# =========================================================

_resolve_exp() {
    local input="$1"

    if [[ "$input" =~ ^[0-9]+$ ]]; then
        local padded
        padded=$(printf "%04d" "$input")

        find "${EXP_ROOT}" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -name "${padded}_*" \
        | head -n1
    else
        echo "${EXP_ROOT}/${input}"
    fi
}

# =========================================================
# Resolve experiment
# =========================================================

EXP_DIR=$(_resolve_exp "$INPUT")
EXP_NAME=$(basename "${EXP_DIR}")
OUT_DIR="${OUT_ROOT}/${EXP_NAME}"
LOG_DIR="${LOG_ROOT}/${EXP_NAME}"

# =========================================================
# Validation
# =========================================================

if [ ! -d "${EXP_DIR}" ]; then
    echo "❌ Experiment directory not found:"
    echo "   ${EXP_DIR}"
    exit 1
fi

# =========================================================
# Parse experiment name
# =========================================================

PARTS=($(echo "${EXP_NAME}" | tr '_' ' '))

ID="${PARTS[0]}"

HAS_DATE=0

if [[ "${PARTS[1]:-}" =~ ^[0-9]{8}$ ]]; then
    HAS_DATE=1
    DATE="${PARTS[1]}"
    PREFIX="${ID}_${DATE}"
else
    PREFIX="${ID}"
fi

# =========================================================
# Preserve FAILED suffix
# =========================================================

if [ "${HAS_DATE}" -eq 1 ]; then
    MAIN_BODY=$(echo "${EXP_NAME}" | sed -E 's/^[0-9]{4}_[0-9]{8}_//')
else
    MAIN_BODY=$(echo "${EXP_NAME}" | sed -E 's/^[0-9]{4}_//')
fi

FAILED_SUFFIX=""

if [[ "${MAIN_BODY}" == *_FAILED_* ]]; then
    FAILED_SUFFIX=$(echo "${MAIN_BODY}" | sed -E 's/^.*(_FAILED_.*)$/\1/')
fi

# =========================================================
# Safe new label
# =========================================================

SAFE_LABEL=$(
    echo "${NEW_LABEL}" \
    | tr ' /' '__' \
    | tr -cd '[:alnum:]_-'
)

NEW_EXP_NAME="${PREFIX}_${SAFE_LABEL}${FAILED_SUFFIX}"

NEW_EXP_DIR="${EXP_ROOT}/${NEW_EXP_NAME}"
NEW_OUT_DIR="${OUT_ROOT}/${NEW_EXP_NAME}"
NEW_LOG_DIR="${LOG_ROOT}/${NEW_EXP_NAME}"

# =========================================================
# Conflict check
# =========================================================

if [ -e "${NEW_EXP_DIR}" ]; then
    echo "❌ Target experiment already exists:"
    echo "   ${NEW_EXP_NAME}"
    exit 1
fi

if [ -e "${NEW_LOG_DIR}" ]; then
    echo "❌ Target log directory already exists:"
    echo "   logs/${NEW_EXP_NAME}"
    exit 1
fi

# =========================================================
# Rename directories
# =========================================================

mv "${EXP_DIR}" "${NEW_EXP_DIR}"

if [ -d "${OUT_DIR}" ]; then
    mv "${OUT_DIR}" "${NEW_OUT_DIR}"
fi

# =========================================================
# Rename logs/
# logs/ を置き去りにすると logs/{旧名} に対応する experiments/{旧名} が
# 無くなるため、make clean_failed が「孤立ログ」として削除してしまう。
# =========================================================

if [ -d "${LOG_DIR}" ]; then
    mv "${LOG_DIR}" "${NEW_LOG_DIR}"

    # logs/{exp}/latest は旧名を含む絶対パスを指しているので張り直す
    if [ -L "${NEW_LOG_DIR}/latest" ]; then
        LATEST_TARGET=$(readlink "${NEW_LOG_DIR}/latest")
        ln -sfn "${NEW_LOG_DIR}/$(basename "${LATEST_TARGET}")" "${NEW_LOG_DIR}/latest"
    fi
fi

# =========================================================
# Rewrite internal references
# =========================================================

RUN_SCRIPT="${NEW_EXP_DIR}/run_slurm.sh"

if [ -f "${RUN_SCRIPT}" ]; then
    sed -i.bak "s|${EXP_NAME}|${NEW_EXP_NAME}|g" "${RUN_SCRIPT}"
    rm -f "${RUN_SCRIPT}.bak"
fi

CONFIG_PATH="${NEW_EXP_DIR}/config.yml"

if [ -f "${CONFIG_PATH}" ]; then
    sed -i.bak "s|${EXP_NAME}|${NEW_EXP_NAME}|g" "${CONFIG_PATH}"
    rm -f "${CONFIG_PATH}.bak"
fi

# =========================================================
# Update latest symlink if needed
# =========================================================

LATEST_LINK="${EXP_ROOT}/latest"

if [ -L "${LATEST_LINK}" ]; then
    CURRENT_TARGET=$(readlink "${LATEST_LINK}")

    if [ "${CURRENT_TARGET}" = "${EXP_DIR}" ] || [ "${CURRENT_TARGET}" = "${EXP_NAME}" ]; then
        ln -sfn "${NEW_EXP_NAME}" "${LATEST_LINK}"
    fi
fi

LATEST_OUT_LINK="${OUT_ROOT}/latest"

if [ -L "${LATEST_OUT_LINK}" ]; then
    CURRENT_TARGET=$(readlink "${LATEST_OUT_LINK}")

    if [ "${CURRENT_TARGET}" = "${OUT_DIR}" ] || [ "${CURRENT_TARGET}" = "${EXP_NAME}" ]; then
        ln -sfn "${NEW_EXP_NAME}" "${LATEST_OUT_LINK}"
    fi
fi

# =========================================================
# Done
# =========================================================

echo ""
echo "✅ Renamed experiment"
echo ""
echo "Old: ${EXP_NAME}"
echo "New: ${NEW_EXP_NAME}"
echo ""
