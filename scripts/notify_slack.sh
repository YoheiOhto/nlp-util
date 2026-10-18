#!/bin/bash

# Notifications are best-effort: they must never change the job result.
_notify_warning() {
    printf 'WARNING: Slack notification skipped: %s\n' "$1" >&2
    return 0
}

# JOB IDから色を生成 (6桁16進数カラーコード)
generate_color_from_id() {
    local job_id="$1"
    # JOB IDをハッシュ化して色に変換
    local hash
    if command -v md5sum >/dev/null 2>&1; then
        hash=$(printf '%s' "$job_id" | md5sum | cut -c1-6) || hash=""
    elif command -v md5 >/dev/null 2>&1; then
        hash=$(printf '%s' "$job_id" | md5 | cut -c1-6) || hash=""
    else
        hash=""
    fi
    if [ -z "$hash" ]; then
        _notify_warning "could not generate a notification color"
        printf '#808080\n'
        return 0
    fi
    echo "#${hash}"
    return 0
}

# Slack通知を送信 (attachments形式で色付き)
notify_slack() {
    local title="$1"
    local emoji="$2"
    local color="$3"
    local fields="$4"
    
    if [ -z "${SLACK_WEBHOOK_URL:-}" ]; then
        return 0
    fi

    if ! command -v jq >/dev/null 2>&1; then
        _notify_warning "jq is not available"
        return 0
    fi

    # JSON構築
    local payload
    if ! payload=$(jq -nc \
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
        }'); then
        _notify_warning "could not build the Slack JSON payload"
        return 0
    fi

    if ! curl --fail --silent --show-error --connect-timeout 5 --max-time 10 -X POST \
        -H 'Content-type: application/json' \
        --data "$payload" \
        "${SLACK_WEBHOOK_URL}" \
        > /dev/null; then
        _notify_warning "webhook request failed"
    fi
    return 0
}

_array_fields() {
    if [ "${RUN_MODE:-single}" != "array" ]; then
        return 0
    fi

    if [[ "${ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]; then
        echo "TASK : $((ARRAY_TASK_ID + 1))/${ARRAY_TOTAL_TASKS:-unknown}"
    else
        echo "TASK : unknown/${ARRAY_TOTAL_TASKS:-unknown}"
    fi
    echo "OPTS : ${ARRAY_TASK_OPTIONS:-unknown}"
    return 0
}

_extra_fields() {
    if [ -n "${ADD_NOTIFY:-}" ]; then
        echo "${ADD_NOTIFY}"
    fi
    return 0
}

_optional_fields() {
    local array_info extra_info

    array_info=$(_array_fields)
    [ -n "${array_info}" ] && printf '%s\n' "${array_info}"

    extra_info=$(_extra_fields)
    [ -n "${extra_info}" ] && printf '%s\n' "${extra_info}"
    return 0
}

notify_start() {
    [ "${SLACK_NOTIFY_ON_START:-1}" = "1" ] || return 0

    local color
    color=$(generate_color_from_id "${JOB_ID:-unknown}")
    local fields="EXP : ${EXP_NAME:-unknown}
JOB : ${JOB_ID:-unknown}
PART : ${JOB_PARTITION:-unknown}
TIME : ${JOB_TIME_LIMIT:-unknown}"

    local optional
    optional=$(_optional_fields)
    [ -n "${optional}" ] && fields="${fields}
${optional}"
    notify_slack "STARTED" "🚀" "$color" "$fields"
    return 0
}

notify_finish() {
    [ "${SLACK_NOTIFY_ON_FINISH:-1}" = "1" ] || return 0

    local color
    color=$(generate_color_from_id "${JOB_ID:-unknown}")
    local fields="EXP : ${EXP_NAME:-unknown}
JOB : ${JOB_ID:-unknown}"

    local optional
    optional=$(_optional_fields)
    [ -n "${optional}" ] && fields="${fields}
${optional}"
    notify_slack "FINISHED" "✅" "$color" "$fields"
    return 0
}

notify_fail() {
    [ "${SLACK_NOTIFY_ON_FAIL:-1}" = "1" ] || return 0

    local color
    color=$(generate_color_from_id "${JOB_ID:-unknown}")
    local fields="EXP : ${EXP_NAME:-unknown}
JOB : ${JOB_ID:-unknown}
STATE : $1"

    local optional
    optional=$(_optional_fields)
    [ -n "${optional}" ] && fields="${fields}
${optional}"
    notify_slack "FAILED" "❌" "$color" "$fields"
    return 0
}

notify_fail_fast() {
    [ "${SLACK_NOTIFY_ON_FAIL:-1}" = "1" ] || return 0

    local color
    color=$(generate_color_from_id "${JOB_ID:-unknown}")
    local fields="EXP : ${EXP_NAME:-unknown}
JOB : ${JOB_ID:-unknown}
STATE : $1"

    local optional
    optional=$(_optional_fields)
    [ -n "${optional}" ] && fields="${fields}
${optional}"
    notify_slack "INTERRUPTED" "⚡" "$color" "$fields"
    return 0
}
