#!/bin/bash

set -euo pipefail

# =====================================================
# Config
# =====================================================

source "${EXP_DIR}/run_slurm.sh"
source "${PROJECT_ROOT}/scripts/notify_slack.sh"

# =====================================================
# Scheduler abstraction
# =====================================================
if [ -n "${PBS_JOBID:-}" ]; then
    export SCHEDULER="pbs"
    export JOB_ID=$(echo "${PBS_JOBID}" | cut -d. -f1)
    
    if [ -n "${PBS_ARRAY_INDEX:-}" ]; then
        export ARRAY_JOB_ID=$(echo "${PBS_JOBID}" | cut -d. -f1 | cut -d'[' -f1)
        export ARRAY_TASK_ID="${PBS_ARRAY_INDEX}"
    else
        export ARRAY_JOB_ID=""
        export ARRAY_TASK_ID=""
    fi
    export JOB_PARTITION="${PBS_QUEUE:-unknown}"
    export NODENAME=$(hostname)
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    export SCHEDULER="slurm"
    export JOB_ID="${SLURM_JOB_ID}"
    export ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-}"
    export ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
    export JOB_PARTITION="${SLURM_JOB_PARTITION:-unknown}"
    export NODENAME="${SLURMD_NODENAME:-unknown}"
else
    export SCHEDULER="local"
    export JOB_ID="$$"
    export ARRAY_JOB_ID=""
    export ARRAY_TASK_ID=""
    export JOB_PARTITION="local"
    export NODENAME=$(hostname)
fi

# =====================================================
# Run mode
#   single : RUN_COMMAND を1つ実行（デフォルト）
#   array  : SLURM_ARRAY_TASK_ID で CONFIGS を引く
#   seq    : CONFIGS をループして順番に実行
# =====================================================

RUN_MODE="${RUN_MODE:-single}"

# =====================================================
# Directories
# =====================================================

# array の場合はジョブIDにarray_idを付与
if [ "${RUN_MODE}" = "array" ]; then
    export JOB_LOG_DIR="${PROJECT_ROOT}/logs/${EXP_NAME}/${ARRAY_JOB_ID}_${ARRAY_TASK_ID}"
else
    export JOB_LOG_DIR="${PROJECT_ROOT}/logs/${EXP_NAME}/${JOB_ID}"
fi

mkdir -p "${JOB_LOG_DIR}"

ln -sfn "${JOB_LOG_DIR}" \
    "${PROJECT_ROOT}/logs/${EXP_NAME}/latest"

export SLURM_LOG_FILE="${JOB_LOG_DIR}/slurm.out"

# =====================================================
# Metadata helper
# =====================================================

update_metadata() {
    local key="$1"
    local value="$2"
    local yaml_file="${JOB_LOG_DIR}/run_metadata.yaml"

    python3 - <<EOF
from pathlib import Path
import yaml

path = Path("${yaml_file}")
data = yaml.safe_load(path.read_text())
data["${key}"] = "${value}"
path.write_text(yaml.safe_dump(data, sort_keys=False))
EOF
}

# =====================================================
# Metadata
# =====================================================

python3 - <<EOF
from pathlib import Path
import yaml

data = {
    "exp_name":   "${EXP_NAME}",
    "job_id":     "${JOB_ID}",
    "partition":  "${JOB_PARTITION}",
    "node":       "${NODENAME}",
    "git_commit": "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)",
    "start_time": "$(date --iso-8601=seconds)",
    "time_limit": "${DEFAULT_TIME}",
    "run_mode":   "${RUN_MODE}",
    "status":     "RUNNING",
}

if "${RUN_MODE}" == "array":
    data["array_task_id"] = "${ARRAY_TASK_ID}"

Path("${JOB_LOG_DIR}/run_metadata.yaml").write_text(
    yaml.safe_dump(data, sort_keys=False)
)
EOF

# The commit hash alone is insufficient when an approved experiment is run
# from a dirty worktree.  Preserve the exact small source/config files that
# define this job, plus hashes and the repository diff visible at job start.
# Generated outputs and large inputs are deliberately not copied here.
SOURCE_SNAPSHOT_DIR="${JOB_LOG_DIR}/source_snapshot"
mkdir -p "${SOURCE_SNAPSHOT_DIR}"
find "${EXP_DIR}" -maxdepth 1 -type f \
    \( -name '*.py' -o -name '*.yml' -o -name '*.yaml' -o -name '*.sh' -o -name '*.md' \) \
    -print0 | sort -z | while IFS= read -r -d '' source_file; do
        cp -- "${source_file}" "${SOURCE_SNAPSHOT_DIR}/$(basename "${source_file}")"
    done
find "${SOURCE_SNAPSHOT_DIR}" -maxdepth 1 -type f -print0 | sort -z | \
    xargs -0 sha256sum > "${JOB_LOG_DIR}/source_snapshot.sha256"
git -C "${PROJECT_ROOT}" status --short > "${JOB_LOG_DIR}/git_status.txt"
git -C "${PROJECT_ROOT}" diff --binary HEAD > "${JOB_LOG_DIR}/git_diff.patch"
if [ "${SCHEDULER}" = "slurm" ] && command -v scontrol >/dev/null 2>&1; then
    scontrol show job "${SLURM_JOB_ID}" > "${JOB_LOG_DIR}/scheduler_job_spec.txt"
fi

# =====================================================
# Move Slurm log helper
# =====================================================

_move_slurm_log() {
    local log_file=""

    if [ "${SCHEDULER}" = "pbs" ]; then
        local pattern=""
        if [ "${RUN_MODE}" = "array" ]; then
            pattern="${PROJECT_ROOT}/logs/${EXP_NAME}/${EXP_NAME}.o${ARRAY_JOB_ID}-${ARRAY_TASK_ID}"
        else
            pattern="${PROJECT_ROOT}/logs/${EXP_NAME}/${EXP_NAME}.o${JOB_ID}"
        fi
        
        log_file=$(ls ${pattern}* 2>/dev/null | head -n1 || true)
        
        if [ -n "${log_file}" ] && [ -f "${log_file}" ]; then
            mv "${log_file}" "${JOB_LOG_DIR}/slurm.out"
        fi
        
        local err_pattern=""
        if [ "${RUN_MODE}" = "array" ]; then
            err_pattern="${PROJECT_ROOT}/logs/${EXP_NAME}/${EXP_NAME}.e${ARRAY_JOB_ID}-${ARRAY_TASK_ID}"
        else
            err_pattern="${PROJECT_ROOT}/logs/${EXP_NAME}/${EXP_NAME}.e${JOB_ID}"
        fi
        local err_log_file
        err_log_file=$(ls ${err_pattern}* 2>/dev/null | head -n1 || true)
        if [ -n "${err_log_file}" ] && [ -f "${err_log_file}" ]; then
            mv "${err_log_file}" "${JOB_LOG_DIR}/pbs.err"
        fi
    else
        if [ "${RUN_MODE}" = "array" ]; then
            # runx が --output=%A_%a_${exp_name}.out で投入するため、
            # %A（親のarray job id = SLURM_ARRAY_JOB_ID）を使う。
            # SLURM_JOB_ID は各タスク固有の値で %A とは異なるため使わない。
            log_file="${PROJECT_ROOT}/logs/${EXP_NAME}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${EXP_NAME}.out"
        else
            log_file="${PROJECT_ROOT}/logs/${EXP_NAME}/${SLURM_JOB_ID}_${EXP_NAME}.out"
        fi

        if [ -f "${log_file}" ]; then
            mv "${log_file}" "${JOB_LOG_DIR}/slurm.out"
        fi
    fi
}

# =====================================================
# sacct helper
# =====================================================

_get_job_state() {
    if [ "${SCHEDULER}" = "pbs" ]; then
        if command -v qstat &>/dev/null; then
            local target_id="${JOB_ID}"
            if [ "${RUN_MODE}" = "array" ]; then
                target_id="${ARRAY_JOB_ID}[${ARRAY_TASK_ID}]"
            fi
            
            local qstat_out
            qstat_out=$(qstat -f "${target_id}" 2>/dev/null || true)
            
            if [ -z "${qstat_out}" ]; then
                echo "COMPLETED"
                return
            fi
            
            local comment
            comment=$(echo "${qstat_out}" | grep -E "comment =" || true)
            
            if echo "${comment}" | grep -iq "walltime"; then
                echo "TIMEOUT"
                return
            fi
            if echo "${comment}" | grep -iq -E "oom|out of memory"; then
                echo "OUT_OF_MEMORY"
                return
            fi
            
            local exit_status
            exit_status=$(echo "${qstat_out}" | grep -E "Exit_status =" | awk '{print $3}' || echo "")
            if [ "${exit_status}" = "271" ]; then
                echo "TIMEOUT"
            elif [ "${exit_status}" = "137" ]; then
                echo "OUT_OF_MEMORY"
            elif [ "${exit_status}" = "0" ] || [ -z "${exit_status}" ]; then
                echo "COMPLETED"
            else
                echo "FAILED"
            fi
        else
            echo "UNKNOWN"
        fi
    else
        if command -v sacct &>/dev/null; then
            sacct \
                -j "${SLURM_JOB_ID}" \
                --format=JobIDRaw,State \
                --parsable2 \
                --noheader \
            | awk -F'|' -v id="${SLURM_JOB_ID}" '$1==id {print $2; exit}'
        elif command -v scontrol &>/dev/null; then
            scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
                | grep -oP 'JobState=\K\w+' || echo "UNKNOWN"
        else
            echo "UNKNOWN"
        fi
    fi
}

_handle_final_state() {
    trap - ERR
    local exit_code="${1:-0}"
    local line_no="${2:-unknown}"

    sleep 3

    local job_state
    job_state=$(_get_job_state || echo "UNKNOWN")

    case "${job_state}" in

        TIMEOUT)
            update_metadata "status" "TIMEOUT"
            update_metadata "fail_reason" "SLURM_TIMEOUT"
            notify_fail "TIMEOUT"
            ;;

        OUT_OF_MEMORY)
            update_metadata "status" "OUT_OF_MEMORY"
            update_metadata "fail_reason" "OUT_OF_MEMORY"
            notify_fail "OUT_OF_MEMORY"
            ;;

        CANCELLED*)
            update_metadata "status" "CANCELLED"
            update_metadata "fail_reason" "CANCELLED"
            notify_fail_fast "CANCELLED"
            ;;

        NODE_FAIL)
            update_metadata "status" "NODE_FAIL"
            update_metadata "fail_reason" "NODE_FAIL"
            notify_fail "NODE_FAIL"
            ;;

        COMPLETED)
            if [ "${exit_code}" -eq 0 ]; then
                update_metadata "status" "COMPLETED"
                notify_finish
            else
                update_metadata "status" "FAILED"
                update_metadata "fail_reason" "NONZERO_EXIT_${exit_code}"
                notify_fail "NONZERO_EXIT (code ${exit_code})"
            fi
            ;;

        RUNNING|COMPLETING|CONFIGURING)
            if [ "${exit_code}" -eq 0 ]; then

                update_metadata "status" "COMPLETED"
                notify_finish

            else

                update_metadata "status" "FAILED"
                update_metadata "fail_reason" "NONZERO_EXIT_${exit_code}"

                notify_fail "NONZERO_EXIT (code ${exit_code})"

            fi
            ;;

        *)
            update_metadata "status" "FAILED"
            update_metadata "fail_reason" "SCRIPT_ERROR_LINE_${line_no}_CODE_${exit_code}"
            notify_fail "SCRIPT_ERROR (line ${line_no}, code ${exit_code}, state ${job_state})"
            ;;

    esac

    _move_slurm_log
}

# =====================================================
# Signal handlers
# =====================================================

on_error() {
    trap - ERR
    local exit_code=$?
    local line_no="${1:-unknown}"

    echo ""
    echo "❌ Error on line ${line_no} (exit code: ${exit_code})"
    echo ""

    _handle_final_state "${exit_code}" "${line_no}"
}

trap 'on_error ${LINENO}' ERR

TERMINATED=0

on_terminate() {
    trap - TERM INT
    TERMINATED=1

    echo "TERM/INT RECEIVED $(date)" \
        >> "${JOB_LOG_DIR}/signal_debug.log"

    _handle_final_state 1 "signal"
}

trap on_terminate TERM INT

# =====================================================
# Time limit warning (旧 watch_job.sh の代替)
# --signal=B:USR1@<margin> により、Slurm が残り時間僅少になった時点で
# このバッチスクリプト自身に USR1 を送る。別プロセスのwatcherは使わない。
# ジョブ自体は継続する（通知のみ、_handle_final_state は呼ばない）。
# =====================================================

on_time_limit_warning() {
    echo "TIME_LIMIT_WARNING RECEIVED $(date)" \
        >> "${JOB_LOG_DIR}/signal_debug.log"
    notify_fail_fast "TIME_LIMIT_WARNING"
}

trap on_time_limit_warning USR1

# =====================================================
# GRID → CONFIGS 展開
# =====================================================

if [ "${RUN_MODE}" != "single" ] \
   && [[ -v GRID_ARGS && "${#GRID_ARGS[@]}" -gt 0 ]]; then

    # GRID_ARGS / GRID_VALUES から直積を生成して CONFIGS に展開
    _expand_grid() {
        local -n _args=$1
        local -n _values=$2
        local -a result=("")

        for i in "${!_args[@]}"; do
            local arg="${_args[$i]}"
            local -a vals
            read -ra vals <<< "${_values[$i]}"

            local -a next=()

            for prev in "${result[@]}"; do
                for val in "${vals[@]}"; do
                    if [ -z "${prev}" ]; then
                        next+=("${arg} ${val}")
                    else
                        next+=("${prev} ${arg} ${val}")
                    fi
                done
            done

            result=("${next[@]}")
        done

        printf '%s\n' "${result[@]}"
    }

    mapfile -t CONFIGS < <(_expand_grid GRID_ARGS GRID_VALUES)
fi

# =====================================================
# Resolve run command
# =====================================================

if [ "${RUN_MODE}" = "array" ]; then

    # CONFIGS[ARRAY_TASK_ID] を引く
    RESOLVED_COMMAND="${BASE_COMMAND} ${CONFIGS[${ARRAY_TASK_ID}]}"

elif [ "${RUN_MODE}" = "seq" ]; then

    # seqの場合はループで実行するのでここではダミー
    RESOLVED_COMMAND=""

else

    RESOLVED_COMMAND="${RUN_COMMAND}"

fi

# =====================================================
# Array info for notifications
# =====================================================

if [ "${RUN_MODE}" = "array" ]; then
    ARRAY_TOTAL_TASKS="${#CONFIGS[@]}"
    ARRAY_TASK_OPTIONS="${CONFIGS[${ARRAY_TASK_ID}]}"
fi

# =====================================================
# Start notify
# =====================================================

notify_start

# =====================================================
# Scratch
# =====================================================

export SCRATCH_DIR="/scratch/${USER}/${EXP_NAME}_${JOB_ID}"

mkdir -p "${SCRATCH_DIR}"

# =====================================================
# Input
# =====================================================

if [ "${USE_LOCAL_SSD_INPUT:-0}" -eq 1 ]; then

    mkdir -p "${SCRATCH_DIR}/data"

    rsync -a \
        "${PROJECT_ROOT}/data/" \
        "${SCRATCH_DIR}/data/"

    export DATASET_DIR="${SCRATCH_DIR}/data"

else

    export DATASET_DIR="${PROJECT_ROOT}/data"
fi

# =====================================================
# Output
# =====================================================

export OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"

mkdir -p "${OUTPUT_DIR}"

# =====================================================
# Save command
# =====================================================

if [ "${RUN_MODE}" = "seq" ]; then
    printf '%s\n' "${CONFIGS[@]/#/${BASE_COMMAND} }" \
        > "${JOB_LOG_DIR}/command.sh"
else
    echo "${RESOLVED_COMMAND}" \
        > "${JOB_LOG_DIR}/command.sh"
fi

# =====================================================
# Run
# =====================================================

_run_single() {
    local cmd="$1"

    set +e

    if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
        apptainer exec \
            --nv \
            --bind "/scratch/${USER}" \
            --bind "${SCRATCH_DIR}" \
            --env UV_CACHE_DIR="${SCRATCH_DIR}/.uv_cache" \
            "${SIF_PATH}" \
            bash -c "
                set -euo pipefail

                source ${PROJECT_ROOT}/.venv/bin/activate
                export CUDA_HOME=/usr/local/cuda

                cd ${PROJECT_ROOT}

                ${cmd}
            " 1>&2 &
    else
        echo "⚠️ Apptainer not found or SIF_PATH not set. Running command on host system." >&2
        bash -c "
            set -euo pipefail
            if [ -f ${PROJECT_ROOT}/.venv/bin/activate ]; then
                source ${PROJECT_ROOT}/.venv/bin/activate
            fi
            cd ${PROJECT_ROOT}
            ${cmd}
        " 1>&2 &
    fi

    local pid=$!
    local code=0

    wait ${pid} || code=$?

    set -e

    echo "${code}"
}

# =====================================================
# Pre-native command (apptainer外で実行したい処理)
# ES 起動など、ネイティブ bash レイヤーで動かすコマンドを
# run_slurm.sh で PRE_NATIVE_COMMAND に設定する
# =====================================================

if [ -n "${PRE_NATIVE_COMMAND:-}" ]; then
    echo "=== PRE_NATIVE_COMMAND ==="
    eval "${PRE_NATIVE_COMMAND}"
fi

EXIT_CODE=0

if [ "${RUN_MODE}" = "seq" ]; then

    # CONFIGS をループして順番に実行
    for i in "${!CONFIGS[@]}"; do

        local_cmd="${BASE_COMMAND} ${CONFIGS[$i]}"

        echo ""
        echo "▶ [${i}/$((${#CONFIGS[@]} - 1))] ${local_cmd}"
        echo ""

        code=$(_run_single "${local_cmd}")

        if [ "${code}" -ne 0 ]; then
            echo "❌ Config [${i}] failed with exit code: ${code}"
            EXIT_CODE="${code}"
            break
        fi
    done

else

    EXIT_CODE=$(_run_single "${RESOLVED_COMMAND}")

fi

if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "Child failed with exit code: ${EXIT_CODE}"
fi

# =====================================================
# Post-native command (apptainer外で実行したい後処理)
# ES データの SSD→HDD 同期など
# =====================================================

if [ -n "${POST_NATIVE_COMMAND:-}" ]; then
    echo "=== POST_NATIVE_COMMAND ==="
    eval "${POST_NATIVE_COMMAND}"
fi

# =====================================================
# Scratch cleanup
# =====================================================

if [ -n "${SCRATCH_DIR:-}" ] \
   && [ -d "${SCRATCH_DIR}" ] \
   && [[ "${SCRATCH_DIR}" == /scratch/* ]]; then

    rm -rf "${SCRATCH_DIR}"
fi

# =====================================================
# Skip final status if already handled by signal
# =====================================================

if [ "${TERMINATED}" -eq 1 ]; then
    exit 1
fi

# =====================================================
# Final status
# =====================================================

_handle_final_state "${EXIT_CODE}" "end"

# Propagate the child status to Slurm.  Without this explicit exit the last
# successful notification/metadata command makes a failed Python child look
# COMPLETED in sacct even though run_metadata.yaml says FAILED.
exit "${EXIT_CODE}"
