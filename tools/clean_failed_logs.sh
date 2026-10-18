#!/bin/bash
# `make clean_failed` / `make failed_log_clean` の実体。
#
# 1. 失敗（FAILED/TIMEOUT/OUT_OF_MEMORY/NODE_FAIL/CANCELLED）したジョブのログを削除
# 2. status が RUNNING のまま放置された「ゾンビ」ログを削除
# 3. 対応する experiments/ が無い孤立ログを削除
#
# ⚠️ 2 について: 以前は run_metadata.yaml の mtime が24時間より古ければ
#    「ハングした RUNNING」と見なして削除していた。しかし run_metadata.yaml は
#    ジョブ起動時に1回書かれたあと終了時まで更新されないため、
#    templates/run_slurm.sh の既定（--time=196:00:00 = 8日）のような長時間ジョブは
#    **稼働中にログディレクトリを消されていた**。消えた後はジョブ終了時の
#    update_metadata も _move_slurm_log も書き込み先を失うため、最終ステータスと
#    slurm.out が丸ごと失われる。
#
#    そのため現在は mtime だけで判断せず、スケジューラ（squeue/qstat）に
#    ジョブがまだ存在するかを必ず確認する。存在する場合、および存在を
#    判定できない場合（スケジューラコマンドが無い等）は削除しない。

set -euo pipefail

PROJECT_ROOT=$(git rev-parse --show-toplevel)
source "${PROJECT_ROOT}/scripts/metadata.sh"

LOGS_DIR="${PROJECT_ROOT}/logs"

if [ ! -d "${LOGS_DIR}" ]; then
    echo "ℹ️  logs/ が存在しないため何もしません。"
    exit 0
fi

# 無更新と見なすまでの時間（分）
STALE_MINUTES=${STALE_MINUTES:-1440}

# =====================================================
# ジョブがまだスケジューラに存在するか
#   0: 存在する（削除してはいけない）
#   1: 存在しない（終了済み）
#   2: 判定できない（安全側に倒して削除しない）
# =====================================================

_job_is_alive() {
    local job_id="$1"
    local out rc

    [ -n "${job_id}" ] || return 2

    if command -v squeue >/dev/null 2>&1; then
        rc=0
        out=$(squeue -h -j "${job_id}" 2>&1) || rc=$?

        if [ "${rc}" -eq 0 ]; then
            if printf '%s' "${out}" | grep -q '[^[:space:]]'; then
                return 0
            fi
            return 1
        fi

        # 「そんなジョブIDは無い」= 終了済み。それ以外（controller不通など）は判定不能
        if printf '%s' "${out}" | grep -qiE 'invalid job id|invalid user'; then
            return 1
        fi
        return 2
    fi

    if command -v qstat >/dev/null 2>&1; then
        if qstat "${job_id}" >/dev/null 2>&1; then
            return 0
        fi
        return 1
    fi

    return 2
}

# =====================================================
# 1 & 2. ジョブ単位のログ
# =====================================================

echo "🔍 Scanning job logs..."

while IFS= read -r meta; do
    [ -n "${meta}" ] || continue
    [ -f "${meta}" ] || continue

    job_dir=$(dirname "${meta}")
    job_label="$(basename "$(dirname "${job_dir}")")/$(basename "${job_dir}")"
    status=$(metadata_get "${meta}" status || true)

    case "${status}" in
        FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|CANCELLED)
            echo "  ❌ Deleting failed job log: ${job_label} (status: ${status})"
            rm -rf "${job_dir}"
            ;;

        RUNNING)
            # まだ新しければ触らない
            if [ -z "$(find "${meta}" -mmin "+${STALE_MINUTES}" 2>/dev/null)" ]; then
                continue
            fi

            job_id=$(metadata_get "${meta}" job_id || true)

            alive_rc=0
            _job_is_alive "${job_id}" || alive_rc=$?

            case "${alive_rc}" in
                0)
                    echo "  ⏭  Keeping ${job_label}: job ${job_id} is still queued/running"
                    ;;
                2)
                    echo "  ⏭  Keeping ${job_label}: could not verify whether job ${job_id:-<unknown>} is still alive"
                    ;;
                *)
                    echo "  ❌ Deleting zombie RUNNING job log: ${job_label} (job ${job_id} is no longer in the scheduler)"
                    rm -rf "${job_dir}"
                    ;;
            esac
            ;;
    esac
done < <(find "${LOGS_DIR}" -name "run_metadata.yaml")

# =====================================================
# 3. 孤立した実験ログ
# =====================================================

echo "🔍 Scanning for orphaned experiment logs..."

while IFS= read -r log_dir; do
    [ -n "${log_dir}" ] || continue

    exp_name=$(basename "${log_dir}")
    [ "${exp_name}" != "latest" ] || continue

    if [ ! -d "${PROJECT_ROOT}/experiments/${exp_name}" ]; then
        echo "  ❌ Deleting orphaned experiment log directory: ${exp_name}"
        rm -rf "${log_dir}"
    fi
done < <(find "${LOGS_DIR}" -mindepth 1 -maxdepth 1 -type d)

echo "✅ Clean up completed."
