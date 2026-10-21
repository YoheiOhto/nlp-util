#!/bin/bash

# JOB IDから色を生成 (6桁16進数カラーコード)
generate_color_from_id() {
    local job_id="$1"
    # JOB IDをハッシュ化して色に変換
    local hash=$(echo -n "$job_id" | md5sum | cut -c1-6)
    echo "#${hash}"
}

# Slack通知を送信 (attachments形式で色付き)
notify_slack() {
    local title="$1"
    local emoji="$2"
    local color="$3"
    local fields="$4"
    
    if [ -z "${SLACK_WEBHOOK_URL:-}" ]; then
        echo "⚠️ Slack notification skipped: SLACK_WEBHOOK_URL is not set." >&2
        return 0
    fi
    
    # エラーでジョブが落ちないように一時的に set +e する
    set +e
    
    # jq が存在しない場合は通知をスキップ
    if ! command -v jq &>/dev/null; then
        echo "⚠️ Slack notification skipped: jq command not found." >&2
        set -e
        return 0
    fi
    
    # JSON構築
    local payload
    local jq_exit_code=0
    payload=$(jq -nc \
        --arg title "$title" \
        --arg emoji "$emoji" \
        --arg color "$color" \
        --arg fields "$fields" \
        '{
            text: ($emoji + " " + $title),
            attachments: [{
                color: $color,
                fields: ($fields | split("\n") | map(split(" : ") | {title: .[0], value: .[1], short: true}))
            }]
        }' 2>/dev/null)
    jq_exit_code=$?
    
    if [ ${jq_exit_code} -ne 0 ] || [ -z "${payload}" ]; then
        echo "⚠️ Slack notification skipped: Failed to build JSON payload via jq (exit code: ${jq_exit_code})." >&2
        set -e
        return 0
    fi
    
    local curl_output
    local curl_exit_code=0
    # curlのエラーも出力にキャプチャするために -sS (silent but show errors) を使う
    curl_output=$(curl -sS -X POST \
        -H 'Content-type: application/json' \
        --data "${payload}" \
        "${SLACK_WEBHOOK_URL}" 2>&1)
    curl_exit_code=$?
    
    if [ ${curl_exit_code} -ne 0 ]; then
        echo "⚠️ Slack notification failed: curl error (exit code: ${curl_exit_code}). Output: ${curl_output}" >&2
    elif [ "${curl_output}" != "ok" ]; then
        echo "⚠️ Slack notification failed: Slack API error. Output: ${curl_output}" >&2
    fi
    
    # 元に戻す
    set -e
}

_array_fields() {
    [ "${RUN_MODE:-}" != "array" ] && return
    echo "TASK : $((SLURM_ARRAY_TASK_ID + 1))/${ARRAY_TOTAL_TASKS}"
    echo "OPTS : ${ARRAY_TASK_OPTIONS}"
}

_extra_fields() {
    if [ -n "${ADD_NOTIFY:-}" ]; then
        echo "${ADD_NOTIFY}"
    fi
}

_append_optional_fields() {
    local -n _fields_ref=$1
    local array_info extra_info

    array_info=$(_array_fields)
    if [ -n "${array_info}" ]; then
        _fields_ref="${_fields_ref}
${array_info}"
    fi

    extra_info=$(_extra_fields)
    if [ -n "${extra_info}" ]; then
        _fields_ref="${_fields_ref}
${extra_info}"
    fi
}

notify_start() {
    [ "${SLACK_NOTIFY_ON_START:-1}" -eq 1 ] || return 0

    local color=$(generate_color_from_id "${SLURM_JOB_ID}")
    local fields="EXP : ${EXP_NAME}
JOB : ${SLURM_JOB_ID}
PART : ${SLURM_JOB_PARTITION}
TIME : ${DEFAULT_TIME}"

    _append_optional_fields fields
    notify_slack "STARTED" "🚀" "$color" "$fields"
}

notify_finish() {
    [ "${SLACK_NOTIFY_ON_FINISH:-1}" -eq 1 ] || return 0

    local color=$(generate_color_from_id "${SLURM_JOB_ID}")
    local fields="EXP : ${EXP_NAME}
JOB : ${SLURM_JOB_ID}"

    _append_optional_fields fields
    notify_slack "FINISHED" "✅" "$color" "$fields"
}

notify_fail() {
    [ "${SLACK_NOTIFY_ON_FAIL:-1}" -eq 1 ] || return 0

    local color=$(generate_color_from_id "${SLURM_JOB_ID}")
    local fields="EXP : ${EXP_NAME}
JOB : ${SLURM_JOB_ID}
STATE : $1"

    _append_optional_fields fields
    notify_slack "FAILED" "❌" "$color" "$fields"
}

notify_fail_fast() {
    [ "${SLACK_NOTIFY_ON_FAIL:-1}" -eq 1 ] || return 0

    local color=$(generate_color_from_id "${SLURM_JOB_ID}")
    local fields="EXP : ${EXP_NAME}
JOB : ${SLURM_JOB_ID}
STATE : $1"

    # SCRATCH_DIR is wiped (rm -rf) unconditionally near the end of
    # slurm_entry.sh, including on this TERM/time-limit-warning path -- if a
    # job is writing unsynced results there (e.g. per-shard checkpoints not
    # yet copied back to shared storage), this is the last chance to retrieve
    # them before the job actually ends.
    if [ -n "${SCRATCH_DIR:-}" ] && [ -d "${SCRATCH_DIR}" ]; then
        fields="${fields}
SCRATCH : ${SCRATCH_DIR} (retrieve now if unsynced -- deleted when the job ends)"
    fi

    _append_optional_fields fields
    notify_slack "INTERRUPTED" "⚡" "$color" "$fields"
}