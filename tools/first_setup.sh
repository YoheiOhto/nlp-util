#!/bin/bash
# `make setup` の実体。何度実行しても同じ結果になる（冪等）。

set -euo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "${PROJECT_ROOT}"

touch .env

# logs/ は #SBATCH --output の出力先なので、これが無いと sbatch 自体が失敗する
mkdir -p logs notebooks data outputs libraries experiments

echo ""
echo "✅ Initial setup completed"
echo ""
echo "Created: .env, logs/, notebooks/, data/, outputs/, libraries/, experiments/"
echo ""
echo "Next steps:"
echo "  1. make build_env p=<partition>   # env/env.def → env/env.sif をビルド"
echo "  2. make uv_sync  p=<partition>    # コンテナ内で uv sync して .venv を作る"
echo "  3. シェルヘルパーを ~/.bashrc から source (README.md 参照)"
echo ""
echo "既にビルド済みのイメージを使う場合は、.env に SIF_PATH=... を書いて 1 を飛ばせます。"
echo ""
