#!/bin/bash
# gitに含まれないデータ（data/ 等）を da1 からリモートへ配布する。
# da1 から実行すること。
#
# Usage:
#   ./libraries/push_to_remote.sh --to miyabi-g --exp 132 --data string biogrid
#   ./libraries/push_to_remote.sh --to miyabi-g --outputs "0131_... 0103_..." --data kegg
#   ./libraries/push_to_remote.sh --to miyabi-g --dry-run --exp 133
#
# オプション:
#   --to miyabi-g|parent|all   転送先（デフォルト: all）
#   --dry-run                  rsync の dry-run
#   --exp <実験番号>            実験ID指定（例: 132 → 0132_* に自動ゼロパディング）
#                              config.yml の upstream を解析して依存 outputs/ を送信リストに追加
#   --data <subdir> [...]      data/ のサブディレクトリを明示指定（複数可）
#   --outputs <name> [...]     outputs/ のディレクトリを明示指定（複数可）
#
# --exp と --outputs は併用可。--data は常に明示指定が必要。

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SRC_REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)

# ============================================================
# ★ 環境設定（新リポジトリへの移植時にここだけ変更する）
# ============================================================
DA1_USER="ohto";          DA1_HOST="192.168.10.31";          DA1_PORT="49152"
DA1_REPO="★/workspace/david0N/ohto/1-projects/..."        # ← 要設定（自身のパス）

PARENT_USER="ohto";       PARENT_HOST="133.11.48.139";        PARENT_PORT="49152"
PARENT_REPO="★/path/to/repo/on/parent"                    # ← 要設定

MIYABI_G_USER="a97006";   MIYABI_G_HOST="miyabi-g.jcahpc.jp"; MIYABI_G_PORT="22"
MIYABI_G_REPO="★/path/to/repo/on/miyabi-g"                # ← 要設定
# ============================================================

# --- 引数パース ---
TO="all"
DRY_RUN_FLAG=""
EXP_IDS=()
DATA_SUBDIRS=()
OUTPUT_NAMES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --to)      TO="$2"; shift ;;
        --dry-run) DRY_RUN_FLAG="--dry-run" ;;
        --exp)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                EXP_IDS+=("$1"); shift
            done
            continue
            ;;
        --data)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DATA_SUBDIRS+=("$1"); shift
            done
            continue
            ;;
        --outputs)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                OUTPUT_NAMES+=("$1"); shift
            done
            continue
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# --- --exp から upstream outputs を解決 ---
resolve_exp_upstreams() {
    local exp_id="$1"
    # 入力を10進数として4桁にゼロパディング（例: 132 → 0132、0132 → 0132）
    local padded
    padded=$(printf "%04d" "$((10#${exp_id}))")
    local exp_dir
    exp_dir=$(find "${SRC_REPO}/experiments" -maxdepth 1 -type d -name "${padded}_*" | head -1)
    if [[ -z "$exp_dir" ]]; then
        echo "  [warn] --exp ${exp_id}: experiments/${padded}_* が見つかりません" >&2
        return
    fi
    local config="${exp_dir}/config.yml"
    if [[ ! -f "$config" ]]; then
        echo "  [warn] --exp ${exp_id}: config.yml が見つかりません (${config})" >&2
        return
    fi
    # upstream: セクションの値（実験名）を抽出
    # 形式: "  key: 0xxx_..." のインデントされた行
    local upstreams
    upstreams=$(awk '/^upstream:/{found=1; next} found && /^[^ ]/{exit} found && /: /{print $NF}' "$config")
    if [[ -z "$upstreams" ]]; then
        echo "  [info] --exp ${exp_id}: upstream なし" >&2
        return
    fi
    echo "$upstreams"
}

declare -A OUTPUTS_SET  # 重複排除用

for exp_id in "${EXP_IDS[@]}"; do
    upstreams=$(resolve_exp_upstreams "$exp_id")
    while IFS= read -r name; do
        [[ -z "$name" ]] && continue
        OUTPUTS_SET["outputs/${name}"]=1
        echo "  [--exp ${exp_id}] → outputs/${name}"
    done <<< "$upstreams"
done

for name in "${OUTPUT_NAMES[@]}"; do
    OUTPUTS_SET["outputs/${name}"]=1
done

# 送信対象リストを確定
SYNC_PATHS=()
for key in "${!OUTPUTS_SET[@]}"; do
    SYNC_PATHS+=("$key")
done
for subdir in "${DATA_SUBDIRS[@]}"; do
    SYNC_PATHS+=("data/${subdir}")
done

if [[ ${#SYNC_PATHS[@]} -eq 0 ]]; then
    echo "エラー: 送信するパスが指定されていません。"
    echo "  --exp <id> / --outputs <name> / --data <subdir> のいずれかを指定してください。"
    exit 1
fi

# --- 確認表示 ---
echo ""
echo "=============================================="
[[ -n "$DRY_RUN_FLAG" ]] && echo " ⚠  DRY RUN"
echo "  src  : ${SRC_REPO}"
echo "  dest : ${TO}"
echo "  送信対象:"
for p in "${SYNC_PATHS[@]}"; do
    local_path="${SRC_REPO}/${p}"
    if [[ -e "$local_path" ]]; then
        size=$(du -sh "$local_path" 2>/dev/null | cut -f1)
        echo "    ${p}/  (${size})"
    else
        echo "    ${p}/  [存在しない → スキップ]"
    fi
done
[[ "$TO" == "miyabi-g" || "$TO" == "all" ]] && echo "  miyabi-g : ${MIYABI_G_USER}@${MIYABI_G_HOST}:${MIYABI_G_REPO}"
[[ "$TO" == "parent"   || "$TO" == "all" ]] && echo "  parent   : ${PARENT_USER}@${PARENT_HOST}:${PARENT_REPO} (pull 方式)"
echo "=============================================="
read -rp "実行しますか？ (y/n): " confirm; echo ""
[[ ! "$confirm" =~ ^[yY]$ ]] && { echo "キャンセルしました。"; exit 1; }

# --- rsync 関数 ---
OPTS="-rlptD --info=progress2 --no-v ${DRY_RUN_FLAG}"

push_miyabi_g() {
    local rel="$1"
    local src="${SRC_REPO}/${rel}"
    local dst="${MIYABI_G_REPO}/${rel}/"
    local dst_str="${MIYABI_G_USER}@${MIYABI_G_HOST}"
    local r_cmd="ssh -p ${MIYABI_G_PORT} -o StrictHostKeyChecking=no"
    if [[ ! -e "$src" ]]; then echo "  miyabi-g  skip : ${rel}/"; return; fi
    echo "  miyabi-g  sync : ${rel}/"
    [[ -z "$DRY_RUN_FLAG" ]] && ssh -p "${MIYABI_G_PORT}" -o StrictHostKeyChecking=no "${dst_str}" "mkdir -p '${dst}'"
    rsync ${OPTS} -e "${r_cmd}" "${src}/" "${dst_str}:${dst}"
}

push_parent() {
    local rel="$1"
    local src="${SRC_REPO}/${rel}"
    local da1_src="${DA1_REPO}/${rel}"
    local parent_dst="${PARENT_REPO}/${rel}/"
    local parent_str="${PARENT_USER}@${PARENT_HOST}"
    local da1_str="${DA1_USER}@${DA1_HOST}"
    if [[ ! -e "$src" ]]; then echo "  parent    skip : ${rel}/"; return; fi
    echo "  parent    sync : ${rel}/"
    local pull_cmd="mkdir -p '${parent_dst}' && rsync ${OPTS} -e 'ssh -p ${DA1_PORT} -o StrictHostKeyChecking=no' '${da1_str}:${da1_src}/' '${parent_dst}'"
    if [[ -n "$DRY_RUN_FLAG" ]]; then
        echo "    [dry] parent 側で実行: ${pull_cmd}"
    else
        ssh -p "${PARENT_PORT}" -o StrictHostKeyChecking=no "${parent_str}" "$pull_cmd"
    fi
}

# --- 実行 ---
for rel in "${SYNC_PATHS[@]}"; do
    case "$TO" in
        miyabi-g) push_miyabi_g "$rel" ;;
        parent)   push_parent   "$rel" ;;
        all)      push_miyabi_g "$rel"; push_parent "$rel" ;;
        *) echo "エラー: 不明な転送先 '$TO'。miyabi-g / parent / all を指定してください。"; exit 1 ;;
    esac
done

echo ""
echo "=============================================="
echo " 完了"
echo "=============================================="
