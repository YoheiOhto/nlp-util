#!/bin/bash
# ==============================================================================
# nlp-envs の SIF を作るジョブ (中身は build.sh)。パーティションは出す人が -p で決める。
# ==============================================================================
# x86_64 (micgm1):
#   cd <nlp-util のフォルダ> && sbatch -p large-micgm1 envs/build.sbatch
# grace (aarch64) 用の SIF は作らない (2026-10-11 に決めた。grace では README の 4.2 の uv で入れる)。
# build.sh の引数 (--envs nlp,llm など) はそのまま後ろに付ける。
# CPU 16・メモリ 64GB・4 時間: llm は wheel だけで数GB あり、SIF への圧縮に時間がかかるため。
# ==============================================================================
#SBATCH --job-name=nlp-envs-build
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=nlp-envs-build-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
echo "host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID}"
bash envs/build.sh "$@"
