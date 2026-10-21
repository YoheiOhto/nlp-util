#!/bin/bash

set -euo pipefail

# =====================================================
# Required env
# =====================================================

: "${PROJECT_ROOT:?}"
: "${EXP_DIR:?}"
: "${EXP_NAME:?}"
: "${SIF_PATH:?}"

# =====================================================
# Imports
# =====================================================

source "${PROJECT_ROOT}/scripts/common.sh"
source "${PROJECT_ROOT}/scripts/notify_slack.sh"

# =====================================================
# Config
# =====================================================

source "${EXP_DIR}/run_slurm.sh"

# =====================================================
# Metadata
# =====================================================

START_TIME=$(date +%s)

# =====================================================
# Output dirs
# =====================================================

export BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"

export JOB_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${SLURM_JOB_ID}"

mkdir -p "${JOB_OUTPUT_DIR}"

ln -sfn \
    "${JOB_OUTPUT_DIR}" \
    "${BASE_OUTPUT_DIR}/latest"

# =====================================================
# Output dir
# =====================================================

if [ "${USE_LOCAL_SSD_OUTPUT}" -eq 1 ]; then

    export OUTPUT_DIR="${SCRATCH_DIR}/output"

else

    export OUTPUT_DIR="${JOB_OUTPUT_DIR}"
fi

mkdir -p "${OUTPUT_DIR}"

# =====================================================
# Scratch
# =====================================================

export SCRATCH_DIR="/scratch/${USER}/${EXP_NAME}_${SLURM_JOB_ID}"

mkdir -p "${SCRATCH_DIR}"

# =====================================================
# Dataset
# =====================================================

if [ "${USE_LOCAL_SSD_INPUT}" -eq 1 ]; then

    export DATASET_DIR="${SCRATCH_DIR}/data"

    mkdir -p "${DATASET_DIR}"

    log_info "Copying dataset to local SSD..."

    rsync -a \
        "${PROJECT_ROOT}/data/" \
        "${DATASET_DIR}/"

else

    export DATASET_DIR="${PROJECT_ROOT}/data"
fi

# =====================================================
# Save metadata
# =====================================================

cat <<EOF > "${JOB_OUTPUT_DIR}/run_metadata.yaml"
exp_name: ${EXP_NAME}
job_id: ${SLURM_JOB_ID}
host: $(hostname)

git_commit: $(git rev-parse HEAD)

start_time: $(date --iso-8601=seconds)

partition: ${SLURM_JOB_PARTITION:-unknown}

output_dir: ${JOB_OUTPUT_DIR}

command: |
$(echo "${RUN_COMMAND}" | sed 's/^/  /')
EOF

echo "${RUN_COMMAND}" \
    > "${JOB_OUTPUT_DIR}/command.sh"

git diff \
    > "${JOB_OUTPUT_DIR}/git_diff.patch"

# =====================================================
# Slack start
# =====================================================

if [ "${ENABLE_SLACK_NOTIFY:-0}" -eq 1 ]; then

    send_slack "
🚀 Experiment started

EXP_NAME: ${EXP_NAME}
JOB_ID: ${SLURM_JOB_ID}
HOST: $(hostname)
"
fi

# =====================================================
# Cleanup
# =====================================================

cleanup() {

    local exit_code=$?

    END_TIME=$(date +%s)

    ELAPSED=$((END_TIME - START_TIME))

    log_info "Cleaning up..."

    # -------------------------------------------------
    # Sync outputs
    # -------------------------------------------------

    if [ "${USE_LOCAL_SSD_OUTPUT}" -eq 1 ]; then

        log_info "Syncing outputs..."

        rsync -a \
            "${OUTPUT_DIR}/" \
            "${JOB_OUTPUT_DIR}/"
    fi

    # -------------------------------------------------
    # Slack
    # -------------------------------------------------

    if [ "${ENABLE_SLACK_NOTIFY:-0}" -eq 1 ]; then

        if [ "$exit_code" -eq 0 ]; then

            send_slack "
✅ Experiment finished

EXP_NAME: ${EXP_NAME}
JOB_ID: ${SLURM_JOB_ID}
TIME: ${ELAPSED}s
OUTPUT: ${JOB_OUTPUT_DIR}
"

        else

            send_slack "
❌ Experiment failed

EXP_NAME: ${EXP_NAME}
JOB_ID: ${SLURM_JOB_ID}
EXIT_CODE: ${exit_code}
OUTPUT: ${JOB_OUTPUT_DIR}
"
        fi
    fi

    # -------------------------------------------------
    # Cleanup scratch
    # -------------------------------------------------

    rm -rf "${SCRATCH_DIR}"

}


# =====================================================
# Runtime info
# =====================================================

log_info "Starting experiment"

echo ""
echo "  EXP_NAME      : ${EXP_NAME}"
echo "  EXP_DIR       : ${EXP_DIR}"
echo "  OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "  DATASET_DIR   : ${DATASET_DIR}"
echo "  SCRATCH_DIR   : ${SCRATCH_DIR}"
echo "  JOB_OUTPUT    : ${JOB_OUTPUT_DIR}"
echo ""

# =====================================================
# Run
# =====================================================

apptainer exec \
    --nv \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "${SCRATCH_DIR}:${SCRATCH_DIR}" \
    "${SIF_PATH}" \
    bash -c "
        set -euo pipefail

        source ${PROJECT_ROOT}/.venv/bin/activate

        cd ${EXP_DIR}

        export OUTPUT_DIR=${OUTPUT_DIR}
        export DATASET_DIR=${DATASET_DIR}

        ${RUN_COMMAND}
    "