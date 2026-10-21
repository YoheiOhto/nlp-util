#!/bin/bash
# Usage:
#   bash tools/create_exp.sh <exp_name>

set -euo pipefail

EXP_NAME=${1:-}
EXAMPLE_NAME=${2:-}

if [ -z "$EXP_NAME" ]; then
    echo "❌ Error: <exp_name> is required."
    echo "Usage: bash tools/create_exp.sh <exp_name> [example_name]"
    exit 1
fi

# =========================================================
# Resolve project root
# =========================================================

PROJECT_ROOT=$(git rev-parse --show-toplevel)

EXP_ROOT="${PROJECT_ROOT}/experiments"
OUT_ROOT="${PROJECT_ROOT}/outputs"

mkdir -p "${EXP_ROOT}"
mkdir -p "${OUT_ROOT}"

# =========================================================
# Generate experiment ID
# =========================================================

LAST_ID=$(
  find "$EXP_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name latest -printf '%f\n' \
  | awk -F_ '/^[0-9]{4}_/ {print $1}' \
  | sort -n \
  | tail -n1
)

if [ -z "${LAST_ID:-}" ]; then
  NEXT_ID="0001"
else
  NEXT_ID=$(printf "%04d" $((10#$LAST_ID + 1)))
fi

if [ -z "${LAST_ID:-}" ]; then
    NEXT_ID=1
else
    NEXT_ID=$((10#$LAST_ID + 1))
fi

EXP_ID=$(printf "%04d" "${NEXT_ID}")

# =========================================================
# Naming
# =========================================================

DATE=$(date +"%Y%m%d")

# Sanitize experiment name
SAFE_EXP_NAME=$(
    echo "$EXP_NAME" \
    | tr ' /' '__' \
    | tr -cd '[:alnum:]_-'
)

DIR_NAME="${EXP_ID}_${DATE}_${SAFE_EXP_NAME}"

EXP_PATH="${EXP_ROOT}/${DIR_NAME}"
OUT_PATH="${OUT_ROOT}/${DIR_NAME}"

# =========================================================
# Safety check
# =========================================================

if [ -e "${EXP_PATH}" ]; then
    echo "❌ Experiment already exists:"
    echo "   ${DIR_NAME}"
    exit 1
fi

# =========================================================
# Create directories
# =========================================================

mkdir -p "${EXP_PATH}"
mkdir -p "${OUT_PATH}"

# =========================================================
# Git info
# =========================================================

COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "git_not_available")
GIT_DIFF=$(git diff HEAD 2>/dev/null || true)

# =========================================================
# Resolve source templates (Default vs Example)
# =========================================================

EXAMPLES_DIR="${PROJECT_ROOT}/examples"
TEMPLATE_DIR="${PROJECT_ROOT}/templates"
SRC_DIR="${TEMPLATE_DIR}"

if [ -n "${EXAMPLE_NAME}" ]; then
    if [ -d "${EXAMPLES_DIR}/${EXAMPLE_NAME}" ]; then
        SRC_DIR="${EXAMPLES_DIR}/${EXAMPLE_NAME}"
        echo "💡 Using specified example: ${EXAMPLE_NAME}"
    else
        echo "❌ Error: Specified example '${EXAMPLE_NAME}' not found in examples/"
        exit 1
    fi
else
    echo "💡 No example specified. Using default templates."
fi

# =========================================================
# Copy and generate files
# =========================================================

# run_slurm.sh (環境に応じたテンプレートからコピー)
PROFILE="${AGENT_PROFILE:-slurm}"
TEMPLATE_FILE="${TEMPLATE_DIR}/run_slurm.sh"
if [ "$PROFILE" = "qsub" ]; then
    TEMPLATE_FILE="${TEMPLATE_DIR}/run_qsub.sh"
elif [ "$PROFILE" = "local" ]; then
    TEMPLATE_FILE="${TEMPLATE_DIR}/run_local.sh"
fi

sed \
    -e "s|__EXP_NAME__|${DIR_NAME}|g" \
    "$TEMPLATE_FILE" \
    > "${EXP_PATH}/run_slurm.sh"

chmod +x "${EXP_PATH}/run_slurm.sh"

# experiment.py
cp \
    "${SRC_DIR}/experiment.py" \
    "${EXP_PATH}/experiment.py"

# config.yml
cp \
    "${SRC_DIR}/config.yml" \
    "${EXP_PATH}/config.yml"

# =========================================================
# Metadata
# =========================================================

cat > "${EXP_PATH}/metadata.yaml" <<EOF
exp_id: ${EXP_ID}
exp_name: ${DIR_NAME}
created_date: ${DATE}
git_commit: ${COMMIT_HASH}
EOF

# =========================================================
# Save uncommitted diff
# =========================================================

if [ -n "${GIT_DIFF}" ]; then
    echo "${GIT_DIFF}" \
        > "${EXP_PATH}/uncommitted_changes.diff"
fi

# =========================================================
# Update latest symlink
# =========================================================

ln -sfn "${EXP_PATH}" "${EXP_ROOT}/latest"
ln -sfn "${OUT_PATH}" "${OUT_ROOT}/latest"

# =========================================================
# Done
# =========================================================

echo ""
echo "✅ Created experiment"
echo ""
echo "ID        : ${EXP_ID}"
echo "Name      : ${DIR_NAME}"
echo "Experiment: ${EXP_PATH}"
echo "Outputs   : ${OUT_PATH}"
echo ""