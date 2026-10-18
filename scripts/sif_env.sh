#!/usr/bin/env bash
# Apptainerイメージ（SIF）の場所を解決する共通ヘルパー。
#
# tools/uv_sync.sh・tools/start_jupyter.sh・scripts/slurm_entry.sh から
# source され、3者が同じ解決規則を使うことを保証する。
#
# 解決順序:
#   1. 既に export されている SIF_PATH（最優先。ジョブ単位の上書き用）
#   2. <project_root>/.env の SIF_PATH=...
#   3. <project_root>/env/env.sif （`make build_env` が生成する既定の場所）
#
# .env は source せず SIF_PATH の行だけを読む（任意のコードを実行しないため）。

_sif_path_from_env_file() {
    local env_file="$1/.env"
    [ -r "${env_file}" ] || return 0

    local line value
    line=$(
        grep -E '^[[:space:]]*(export[[:space:]]+)?SIF_PATH[[:space:]]*=' "${env_file}" \
        | tail -n1 \
        || true
    )
    [ -n "${line}" ] || return 0

    value=${line#*=}
    value=$(
        printf '%s' "${value}" \
        | sed -e 's/^[[:space:]]*//' \
              -e 's/[[:space:]]*$//' \
              -e 's/^"\(.*\)"$/\1/' \
              -e "s/^'\(.*\)'\$/\1/"
    )

    # source しない代わりに、先頭の ~ だけは自前で展開する
    case "${value}" in
        "~"/*) value="${HOME}/${value#\~/}" ;;
    esac

    printf '%s' "${value}"
}

# SIF_PATH を解決して export する（存在検証はしない）。
resolve_sif_path() {
    local project_root="$1"

    if [ -n "${SIF_PATH:-}" ]; then
        export SIF_PATH
        return 0
    fi

    local from_env_file
    from_env_file=$(_sif_path_from_env_file "${project_root}")

    if [ -n "${from_env_file}" ]; then
        export SIF_PATH="${from_env_file}"
    else
        export SIF_PATH="${project_root}/env/env.sif"
    fi

    return 0
}

# SIF_PATH を解決し、apptainer とイメージが実際に使える状態かを検証する。
# 使えない場合は次に何をすべきかを示して非ゼロで返る。
require_sif() {
    local project_root="$1"
    resolve_sif_path "${project_root}"

    if ! command -v apptainer >/dev/null 2>&1; then
        printf '❌ apptainer が見つかりません (host: %s)\n' "$(hostname)" >&2
        printf '   このテンプレートは全ジョブをApptainerイメージ内で実行するため必須です。\n' >&2
        printf '   ログインノードではなく計算ノード上で実行されているか確認してください。\n' >&2
        return 1
    fi

    if [ ! -f "${SIF_PATH}" ]; then
        printf '❌ Apptainerイメージが見つかりません: %s\n' "${SIF_PATH}" >&2
        printf '   先にビルドしてください:  make build_env p=<partition>\n' >&2
        printf '   既存のイメージを使う場合は %s/.env に SIF_PATH=... を書いてください。\n' \
            "${project_root}" >&2
        return 1
    fi

    return 0
}
