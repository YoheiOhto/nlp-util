#!/bin/bash
#SBATCH --job-name=build_env
#SBATCH --time=01:00:00
#SBATCH --output=logs/build_env_%j.out
#SBATCH --error=logs/build_env_%j.out
#
# env/env.def から env/env.sif をビルドする。
#
# partition はこのファイルに焼き込まず、`make build_env p=<partition>` が
# `sbatch --partition=<p>` として渡す（tools/uv_sync.sh・tools/start_jupyter.sh
# と同じ方式。マシン・アカウントごとに変わる値をリポジトリに固定しないため）。

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${PROJECT_ROOT}"

DEF_FILE="${PROJECT_ROOT}/env/env.def"
SIF_FILE="${PROJECT_ROOT}/env/env.sif"

# ビルド失敗時に既存の動くイメージを壊さないため、一旦別名で作って成功後に差し替える
NEW_SIF="${SIF_FILE}.new"

if ! command -v apptainer >/dev/null 2>&1; then
    echo "❌ apptainer が見つかりません (host: $(hostname))" >&2
    echo "   ログインノードではなく計算ノードに投入されているか確認してください。" >&2
    exit 1
fi

if [ ! -f "${DEF_FILE}" ]; then
    echo "❌ definition file が見つかりません: ${DEF_FILE}" >&2
    exit 1
fi

# =====================================================
# Build scratch space
# CUDA devel ベースのイメージは展開時に数十GBの一時領域を使う。
# デフォルトの /tmp（ノードによってはtmpfs=RAM）では溢れるため、
# ノード付属SSDの /scratch を優先して使う。
# =====================================================

if [ -d /scratch ] && mkdir -p "/scratch/${USER}/apptainer_build" 2>/dev/null; then
    BUILD_SCRATCH="/scratch/${USER}/apptainer_build"
else
    BUILD_SCRATCH="${TMPDIR:-/tmp}/apptainer_build_${USER}"
    mkdir -p "${BUILD_SCRATCH}"
fi

mkdir -p "${BUILD_SCRATCH}/tmp" "${BUILD_SCRATCH}/cache"
export APPTAINER_TMPDIR="${BUILD_SCRATCH}/tmp"
export APPTAINER_CACHEDIR="${BUILD_SCRATCH}/cache"

echo "=========================================="
echo "Building Apptainer image"
echo "  def       : ${DEF_FILE}"
echo "  sif       : ${SIF_FILE}"
echo "  tmpdir    : ${APPTAINER_TMPDIR}"
echo "  cachedir  : ${APPTAINER_CACHEDIR}"
echo "  node      : $(hostname)"
echo "=========================================="

if apptainer build --force "${NEW_SIF}" "${DEF_FILE}"; then
    mv -f "${NEW_SIF}" "${SIF_FILE}"
    echo "=========================================="
    echo "Build successful!"
    echo "  ${SIF_FILE}"
    echo ""
    echo "Next: make uv_sync p=<partition>"
    echo "=========================================="
else
    rm -f "${NEW_SIF}"
    echo "Build failed..." >&2
    exit 1
fi
