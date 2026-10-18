#!/usr/bin/env bash
# logs/{exp_name}/{job_id}/run_metadata.yaml の読み書きヘルパー。
#
# scripts/slurm_entry.sh と tools/cancel_job.sh から source される。
#
# このファイルは「フラットなスカラーのマップ」しか扱わない。run_metadata.yaml に
# ネストや配列が入ることはないため、YAMLパーサ（PyYAML）は使わない:
#   - 以前は python3 + yaml.safe_dump で書く経路と printf で直書きする経路
#     （_bootstrap_fail）が混在していた
#   - 読み手は元々 grep（Makefile の failed_log_clean）
#   - PyYAML はホストの python3 に入っている保証がなく、ジョブのライフサイクル
#     管理がその有無に依存していた。シグナルハンドラ(TERM/INT)の中で python
#     サブプロセスを起動する形にもなっていた
# 値は常にダブルクォートで囲んで書くので、`:` や `#` を含む fail_reason でも
# YAML として妥当なまま保たれる。

# 値を YAML のダブルクォート文字列にする
_metadata_quote() {
    local v="$1"
    v=${v//\\/\\\\}
    v=${v//\"/\\\"}
    printf '"%s"' "${v}"
}

# ダブルクォート文字列を元に戻す
_metadata_unquote() {
    local v="$1"
    # 先頭の空白を落とす
    v="${v#"${v%%[![:space:]]*}"}"
    if [ "${#v}" -ge 2 ] && [ "${v:0:1}" = '"' ] && [ "${v: -1}" = '"' ]; then
        v="${v:1:${#v}-2}"
        v=${v//\\\"/\"}
        v=${v//\\\\/\\}
    fi
    printf '%s' "${v}"
}

# metadata_init <file> <key> <value> [<key> <value> ...]
# ファイルを新規作成（既存内容は破棄）する。
metadata_init() {
    local file="$1"; shift
    local tmp="${file}.tmp.$$"

    : > "${tmp}" || return 1
    while [ "$#" -ge 2 ]; do
        printf '%s: %s\n' "$1" "$(_metadata_quote "$2")" >> "${tmp}" || return 1
        shift 2
    done
    if [ "$#" -ne 0 ]; then
        printf 'metadata_init: odd number of key/value arguments\n' >&2
        rm -f "${tmp}"
        return 1
    fi

    mv -f "${tmp}" "${file}"
}

# metadata_set <file> <key> <value>
# キーがあれば置換、なければ追記する（順序は保たれる）。
metadata_set() {
    local file="$1" key="$2" value="$3"
    local tmp="${file}.tmp.$$"
    local found=0 line

    : > "${tmp}" || return 1

    if [ -f "${file}" ]; then
        while IFS= read -r line || [ -n "${line}" ]; do
            case "${line}" in
                "${key}:"|"${key}: "*)
                    printf '%s: %s\n' "${key}" "$(_metadata_quote "${value}")" >> "${tmp}" || return 1
                    found=1
                    ;;
                *)
                    printf '%s\n' "${line}" >> "${tmp}" || return 1
                    ;;
            esac
        done < "${file}"
    fi

    if [ "${found}" -eq 0 ]; then
        printf '%s: %s\n' "${key}" "$(_metadata_quote "${value}")" >> "${tmp}" || return 1
    fi

    mv -f "${tmp}" "${file}"
}

# metadata_get <file> <key>
# 値を標準出力に返す。キーが無ければ非ゼロで返る。
metadata_get() {
    local file="$1" key="$2" line
    [ -r "${file}" ] || return 1

    line=$(grep -m1 -E "^${key}:" "${file}" || true)
    [ -n "${line}" ] || return 1

    _metadata_unquote "${line#*:}"
}
