#!/bin/bash

set -euo pipefail

JOB_ID=${1:-}
REASON=${2:-"user_cancel"}

if [ -z "${JOB_ID}" ]; then
    echo "Usage:"
    echo "  cancel_job.sh <job_id> [reason]"
    echo "  cancel_job.sh --all [reason]"
    exit 1
fi

# =====================================================
# Root
# =====================================================

PROJECT_ROOT=$(git rev-parse --show-toplevel)

# run_metadata.yaml の読み書き（scripts/slurm_entry.sh と同じ実装を共有する）
source "${PROJECT_ROOT}/scripts/metadata.sh"

# =====================================================
# --all: cancel every job currently queued/running for $USER
# Resolves each distinct base job ID from squeue (array task suffixes like
# "_12" stripped) and re-invokes this script once per job ID, so the existing
# single-job path below (metadata update + Slack notify + scancel) is reused
# as-is instead of being duplicated.
# =====================================================

if [ "${JOB_ID}" = "--all" ]; then
    if ! command -v squeue &>/dev/null; then
        echo "❌ squeue not found — cannot resolve jobs for --all"
        exit 1
    fi

    JOB_IDS=$(squeue -u "${USER}" -h -o "%A" | sed -E 's/_.*//' | sort -u || true)

    if [ -z "${JOB_IDS}" ]; then
        echo "No running/pending jobs found for user ${USER}."
        exit 0
    fi

    echo "Found jobs for ${USER}: ${JOB_IDS}"

    for JID in ${JOB_IDS}; do
        echo ""
        echo "=== Cancelling job ${JID} ==="
        # 1件失敗しても残りのキャンセルは続ける
        bash "${PROJECT_ROOT}/tools/cancel_job.sh" "${JID}" "${REASON}" \
            || echo "⚠️  job ${JID} のキャンセルに失敗しました（続行します）" >&2
    done

    exit 0
fi

# =====================================================
# Resolve metadata
# Array job の場合 logs 下に {array_job_id}_{array_task_id}/ というディレクトリが
# できるため、「完全一致」と「{job_id}_ で始まるもの」の2パターンを探す。
#
# ⚠️ 以前は -path "*/${JOB_ID}*/..." という前方一致だったため、
#    `cancelx 1234` が 12345 や 123456 といった無関係な稼働中ジョブの
#    メタデータまで CANCELLED に書き換えていた（scancel は 1234 しか止めない
#    ので、走り続けているジョブのステータスだけが壊れる）。
# =====================================================

META_FILES=$(
    find "${PROJECT_ROOT}/logs" \
        \( -path "*/${JOB_ID}/run_metadata.yaml" \
           -o -path "*/${JOB_ID}_*/run_metadata.yaml" \) \
        2>/dev/null \
    || true
)

if [ -z "${META_FILES}" ]; then
    echo "⚠️  Could not find metadata for job ${JOB_ID}"
    echo "    Proceeding with scancel only."
else
    # =====================================================
    # Update metadata for all matching tasks
    # =====================================================

    while IFS= read -r META_FILE; do

        echo "Updating: ${META_FILE}"

        metadata_set "${META_FILE}" status        "CANCELLED"
        metadata_set "${META_FILE}" cancel_reason "${REASON}"
        metadata_set "${META_FILE}" cancelled_by  "${USER}"

    done <<< "${META_FILES}"
fi

# =====================================================
# Parse exp name for Slack
# (最初に見つかったメタデータから取得)
# =====================================================

FIRST_META=$(echo "${META_FILES}" | head -n1)

EXP_NAME=""

if [ -n "${FIRST_META}" ]; then
    EXP_NAME=$(metadata_get "${FIRST_META}" exp_name || true)
fi

# =====================================================
# Slack notify
# =====================================================

source "${PROJECT_ROOT}/scripts/notify_slack.sh"

export EXP_NAME="${EXP_NAME}"
export SLURM_JOB_ID="${JOB_ID}"

notify_fail_fast "USER_CANCEL (reason: ${REASON}, by: ${USER})"

# =====================================================
# Cancel
# =====================================================

echo ""
echo "🛑 Cancelling job ${JOB_ID}"
echo ""

if command -v scancel &>/dev/null; then
    scancel "${JOB_ID}"
elif command -v qdel &>/dev/null; then
    # アレイジョブ対応：PBSではアレイジョブ全体を削除する場合、IDのみで指定可能
    qdel "${JOB_ID}"
elif kill -0 "${JOB_ID}" 2>/dev/null; then
    echo "⚠️ No scheduler command found. Killing local process (PID: ${JOB_ID})"
    kill "${JOB_ID}"
else
    echo "❌ No active job or scheduler cancel command found (scancel/qdel/local pid)"
    exit 1
fi

echo "✅ Done"
