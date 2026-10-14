#!/bin/bash
# ==============================================================================
# nlp-envs の SIF を GPU で確かめるジョブ (中身は test.sh)。パーティションは出す人が -p で決める。
# ==============================================================================
# x86_64 (micgm1):  sbatch -p small-micgm1 envs/test.sbatch
# aarch64 (grace02): sbatch -p small-grace-o -w grace02 envs/test.sbatch --out /workspace/filesrv01/$USER/nlp-envs/images
# vLLM で実際に生成も試すなら、後ろに --vllm-model Qwen/Qwen3-0.6B など (モデルをダウンロードする)。
# ==============================================================================
#SBATCH --job-name=nlp-envs-test
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=nlp-envs-test-%j.log

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
bash envs/test.sh "$@"
