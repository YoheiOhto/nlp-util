#!/usr/bin/env bash
# ==============================================================================
# 作った nlp-envs の SIF を、この機械の GPU で確かめる (中身は tests/smoke.py と es/smoke_es.sh)
# ==============================================================================
# 使い方 (ふつうは test.sbatch から GPU のジョブとして流す):
#   bash envs/test.sh [--envs nlp,llm,train,es] [--out DIR] [--vllm-model <Hugging Face のモデル名>]
#   --vllm-model を付けると、llm でそのモデルを読み込んで1文を生成する (モデルをダウンロードする)。
# 結果は <名前>.test.log に残し、最後に PASS / FAIL の一覧を出す。1つでも FAIL なら終了コード 1。
# ==============================================================================
set -uo pipefail

ENVS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
want_envs="nlp,llm,train,es"
out_dir="${NLP_ENVS_IMAGES:-$(cd "$ENVS_DIR/../.." && pwd)/images}"
vllm_model=""
while [ $# -gt 0 ]; do
    case "$1" in
        --envs) want_envs="$2"; shift 2 ;;
        --out) out_dir="$2"; shift 2 ;;
        --vllm-model) vllm_model="$2"; shift 2 ;;
        -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

case "$(uname -m)" in
    x86_64) arch=x86_64 ;;
    aarch64|arm64) arch=aarch64 ;;
    *) echo "unsupported CPU $(uname -m)" >&2; exit 1 ;;
esac
version="$(python3 -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1],"rb"))["project"]["version"])' "$ENVS_DIR/pyproject.toml")"

# 利用者のシェルの uv・Python の設定 (PYTHONPATH など) が入ると SIF の外のパッケージを読みうるので消す (build.sh と同じ)
for name in $(compgen -e); do
    case "$name" in UV_*|PYTHON*|VIRTUAL_ENV|CONDA_*|PIP_*) unset "$name" ;; esac
done

failed=0
summary=()
for env in ${want_envs//,/ }; do
    name="${env}-${version}-${arch}"
    sif="$out_dir/${name}.sif"
    log="$out_dir/${name}.test.log"
    if [ ! -e "$sif" ]; then
        summary+=("FAIL ${name}: ${sif} not found")
        failed=1
        continue
    fi
    echo "== testing ${name}"
    if [ "$env" = es ]; then
        bash "$ENVS_DIR/es/smoke_es.sh" "$sif" > "$log" 2>&1
    else
        args=(--env "$env")
        [ "$env" = llm ] && [ -n "$vllm_model" ] && args+=(--vllm-model "$vllm_model")
        apptainer exec --nv "$sif" python /opt/nlp-envs/smoke.py "${args[@]}" > "$log" 2>&1
    fi
    status=$?
    cat "$log"
    if [ "$status" -eq 0 ]; then
        summary+=("PASS ${name} (log ${log})")
    else
        summary+=("FAIL ${name} (exit ${status}, log ${log})")
        failed=1
    fi
done

echo "== summary on $(hostname) ($(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1))"
printf '%s\n' "${summary[@]}"
exit "$failed"
