#!/bin/bash
# リモートから da1 へ logs/outputs/experiments/data/results を集約する。
# da1 のファイルは削除しない。
#
# Usage:
#   ./libraries/collect_from_remote.sh [--from parent|miyabi-g] [--dry-run]
#
# --from parent   (default) parent から実行すること — ローカル NFS → da1 SSH push
# --from miyabi-g           da1 から実行すること   — miyabi-g SSH → ローカル pull

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
LOCAL_REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)

# ============================================================
# ★ 環境設定（新リポジトリへの移植時にここだけ変更する）
# ============================================================
DA1_USER="ohto";          DA1_HOST="192.168.10.31";      DA1_PORT="49152"
DA1_REPO="★/workspace/david0N/ohto/1-projects/..."   # ← 要設定

MIYABI_G_USER="a97006";   MIYABI_G_HOST="miyabi-g.jcahpc.jp"; MIYABI_G_PORT="22"
MIYABI_G_REPO="★/path/to/repo/on/miyabi-g"           # ← 要設定

COLLECT_DIRS=(logs outputs results experiments data)
# ============================================================

FROM="parent"
DRY_RUN_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)    FROM="$2"; shift ;;
        --dry-run) DRY_RUN_FLAG="--dry-run" ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

OPTS="-rlptD --info=progress2 --no-v ${DRY_RUN_FLAG}"
SYNCED=0; SKIPPED=0

case "$FROM" in
parent)
    # parent から実行: SRC=ローカル(parent NFS), DST=da1 via SSH
    D_STR="${DA1_USER}@${DA1_HOST}"
    R_CMD="ssh -p ${DA1_PORT} -o StrictHostKeyChecking=no"
    echo ""
    echo "=============================================="
    [[ -n "$DRY_RUN_FLAG" ]] && echo " ⚠  DRY RUN"
    echo "  parent  →  da1"
    echo "  src : ${LOCAL_REPO}"
    echo "  dst : ${D_STR}:${DA1_REPO}"
    echo "  dirs: ${COLLECT_DIRS[*]}"
    echo "=============================================="
    read -rp "実行しますか？ (y/n): " confirm; echo ""
    [[ ! "$confirm" =~ ^[yY]$ ]] && { echo "キャンセルしました。"; exit 1; }

    for dir in "${COLLECT_DIRS[@]}"; do
        src="${LOCAL_REPO}/${dir}"; dst="${DA1_REPO}/${dir}/"
        if [[ ! -d "$src" ]]; then
            echo "  skip : ${dir}/ (ローカルに存在しない)"
            SKIPPED=$((SKIPPED+1)); continue
        fi
        echo "  sync : ${dir}/"
        [[ -z "$DRY_RUN_FLAG" ]] && ssh -p "${DA1_PORT}" -o StrictHostKeyChecking=no "${D_STR}" "mkdir -p '${dst}'"
        rsync ${OPTS} -e "${R_CMD}" "${src}/" "${D_STR}:${dst}"
        SYNCED=$((SYNCED+1))
    done
    ;;

miyabi-g)
    # da1 から実行: SRC=miyabi-g via SSH, DST=ローカル(da1)
    S_STR="${MIYABI_G_USER}@${MIYABI_G_HOST}"
    R_CMD="ssh -p ${MIYABI_G_PORT} -o StrictHostKeyChecking=no"
    echo ""
    echo "=============================================="
    [[ -n "$DRY_RUN_FLAG" ]] && echo " ⚠  DRY RUN"
    echo "  miyabi-g  →  da1 (local)"
    echo "  src : ${S_STR}:${MIYABI_G_REPO}"
    echo "  dst : ${LOCAL_REPO}"
    echo "  dirs: ${COLLECT_DIRS[*]}"
    echo "=============================================="
    read -rp "実行しますか？ (y/n): " confirm; echo ""
    [[ ! "$confirm" =~ ^[yY]$ ]] && { echo "キャンセルしました。"; exit 1; }

    for dir in "${COLLECT_DIRS[@]}"; do
        src="${MIYABI_G_REPO}/${dir}"; dst="${LOCAL_REPO}/${dir}/"
        if ! ssh -p "${MIYABI_G_PORT}" -o StrictHostKeyChecking=no "${S_STR}" "[ -d '${src}' ]" 2>/dev/null; then
            echo "  skip : ${dir}/ (転送元に存在しない)"
            SKIPPED=$((SKIPPED+1)); continue
        fi
        echo "  sync : ${dir}/"
        mkdir -p "${dst}"
        rsync ${OPTS} -e "${R_CMD}" "${S_STR}:${src}/" "${dst}"
        SYNCED=$((SYNCED+1))
    done
    ;;

*)
    echo "エラー: 不明なマシン '$FROM'。parent / miyabi-g を指定してください。"
    exit 1
    ;;
esac

echo ""
echo "=============================================="
echo " 完了: ${SYNCED} 件同期, ${SKIPPED} 件スキップ"
echo "=============================================="
