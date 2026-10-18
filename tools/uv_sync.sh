#!/bin/bash
#SBATCH --job-name=uv_sync
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --output=logs/uv-sync-log_%j.out
#SBATCH --error=logs/uv-sync-log_%j.out
#
# partition は `make uv_sync p=<partition>` が sbatch 側で渡す。
#
# 追加の extra は位置引数で受ける:
#   make uv_sync p=<partition> extras="llm chem"   → uv sync --extra llm --extra chem
#   make uv_sync p=<partition> extras=all          → uv sync --all-extras
# 引数なしなら base のみ（pyproject.toml の [project.dependencies]）。

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${PROJECT_ROOT}"

# =====================================================
# Extras
# このあと apptainer 内の `bash -c "..."` 文字列に埋め込むため、
# extra 名は英数字・アンダースコア・ハイフンのみに制限する。
# =====================================================

SYNC_FLAGS=""
EXTRA_LABEL="(base only)"

if [ "$#" -gt 0 ]; then
    for e in "$@"; do
        if ! printf '%s' "${e}" | grep -qE '^[A-Za-z0-9_-]+$'; then
            echo "❌ 不正な extra 名: ${e}" >&2
            echo "   英数字・アンダースコア・ハイフンのみ使用できます。" >&2
            exit 1
        fi
        if [ "${e}" = "all" ]; then
            SYNC_FLAGS="${SYNC_FLAGS} --all-extras"
        else
            SYNC_FLAGS="${SYNC_FLAGS} --extra ${e}"
        fi
    done
    EXTRA_LABEL="$*"
fi

# SIF_PATH の解決（export済み → .env → env/env.sif）と存在検証。
# ここで落とさないと「uv sync completed successfully」だけ出て .venv が
# 作られていない、という偽の成功になる。
source "${PROJECT_ROOT}/scripts/sif_env.sh"
require_sif "${PROJECT_ROOT}"


echo "=========================================="
echo "Starting uv sync on compute node..."
echo "  Project Root : ${PROJECT_ROOT}"
echo "  SIF          : ${SIF_PATH}"
echo "  Node         : $(hostname)"
echo "  Extras       : ${EXTRA_LABEL}"
echo "=========================================="

# Execute uv sync inside the Apptainer container
# Note: 'cd' into PROJECT_ROOT inside the container to ensure .venv is created in the right place
# The exit code of `uv sync` is checked explicitly below: without this, a failing
# `uv sync` would still let the script reach the final echo and exit 0, making
# Slurm (and `make uv_sync`) report the job as COMPLETED even though the venv
# was never synced.
if apptainer exec \
    --nv \
    --bind "${PROJECT_ROOT}" \
    "${SIF_PATH}" \
    bash -c "
        set -euo pipefail
        cd ${PROJECT_ROOT}
        uv sync${SYNC_FLAGS}
    "; then
    echo "=========================================="
    echo "uv sync completed successfully."
    echo "  venv: ${PROJECT_ROOT}/.venv"
    echo "=========================================="
else
    exit_code=$?
    echo "==========================================" >&2
    echo "❌ uv sync failed (exit code: ${exit_code})." >&2
    echo "==========================================" >&2
    exit "${exit_code}"
fi
