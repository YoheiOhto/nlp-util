#!/usr/bin/env bash
# ==============================================================================
# nlp-envs の SIF を、この機械の CPU (x86_64 / aarch64) 用に作る
# ==============================================================================
# 使い方 (ふつうは build.sbatch から Slurm のジョブとして流す。README.md の「作る」):
#   bash envs/build.sh [--envs nlp,llm,train,es] [--out DIR] [--work DIR] [--allow-dirty]
#
#   --envs         作る環境 (既定: 全部)。土台 (base) は必要なら先に作る
#   --out          SIF を置く場所 (既定: $NLP_ENVS_IMAGES、無ければリポジトリの外の ../../images)
#   --work         作業用の一時ファイル (数十GB)。既定は /scratch/$USER/nlp-envs-build、無ければ <out>/.work
#   --allow-dirty  コミットしていない変更があっても作る (試しだけ。名前に -dirty が付き、公開には使わない)
#
# 決まり:
#   - 名前は <env>-<version>-<arch>.sif (例 llm-2026.10.0-aarch64.sif)。version は pyproject.toml の project.version
#   - 作った SIF は上書きしない (同じ名前があれば止まる)。中身を変えたら version を上げる
#   - 横に <名前>.sha256 と <名前>.json (版・コミット・lock のハッシュなど) を残す
# ==============================================================================
set -euo pipefail

ENVS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALL_ENVS="nlp,llm,train,es"
want_envs="$ALL_ENVS"
out_dir="${NLP_ENVS_IMAGES:-$(cd "$ENVS_DIR/../.." && pwd)/images}"
work_dir=""
allow_dirty=0

while [ $# -gt 0 ]; do
    case "$1" in
        --envs) want_envs="$2"; shift 2 ;;
        --out) out_dir="$2"; shift 2 ;;
        --work) work_dir="$2"; shift 2 ;;
        --allow-dirty) allow_dirty=1; shift ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

# apptainer は外の環境変数をコンテナに渡す。利用者のシェルの uv・Python の設定 (UV_CACHE_DIR など) が
# 入ると、作るときの確認がコンテナから見えない場所を使おうとして止まり、中身も変わりうるので消す。
for name in $(compgen -e); do
    case "$name" in UV_*|PYTHON*|VIRTUAL_ENV|CONDA_*|PIP_*) unset "$name" ;; esac
done

case "$(uname -m)" in
    x86_64) arch=x86_64 ;;
    aarch64|arm64) arch=aarch64 ;;
    *) die "unsupported CPU $(uname -m) (x86_64 or aarch64 only)" ;;
esac
command -v apptainer >/dev/null 2>&1 || die "apptainer is not installed on $(hostname)"
command -v python3 >/dev/null 2>&1 || die "python3 is needed to read pyproject.toml"

# pyproject.toml から版を読む (ここが唯一の置き場所)
read_config() {
    python3 - "$ENVS_DIR/pyproject.toml" "$1" <<'PY'
import sys, tomllib
data = tomllib.load(open(sys.argv[1], "rb"))
key = sys.argv[2]
value = data["project"]["version"] if key == "version" else data["tool"]["nlp-envs"][key]
print(value)
PY
}
version="$(read_config version)"
cuda_tag="$(read_config cuda-image-tag)"
uv_version="$(read_config uv-version)"
python_version="$(read_config python-version)"
es_version="$(read_config es-version)"

# どのコミットから作ったか。変更が残っていたら止める (再現できる SIF だけを公開するため)
git_commit="$(git -C "$ENVS_DIR" rev-parse --short=12 HEAD)"
if [ -n "$(git -C "$ENVS_DIR" status --porcelain -- .)" ]; then
    [ "$allow_dirty" -eq 1 ] || die "uncommitted changes under envs/. Commit first, or use --allow-dirty for a trial build."
    git_commit="${git_commit}-dirty"
    version="${version}-dirty"
fi

mkdir -p "$out_dir"
if [ -z "$work_dir" ]; then
    if [ -d /scratch ] && mkdir -p "/scratch/${USER}/nlp-envs-build" 2>/dev/null; then
        work_dir="/scratch/${USER}/nlp-envs-build"
    else
        work_dir="$out_dir/.work"
    fi
fi
mkdir -p "$work_dir/tmp" "$work_dir/cache" "$work_dir/uv-cache"
export APPTAINER_TMPDIR="$work_dir/tmp"
export APPTAINER_CACHEDIR="$work_dir/cache"

lock_sha="$(sha256sum "$ENVS_DIR/uv.lock" | cut -d' ' -f1)"
echo "== nlp-envs ${version} for ${arch} on $(hostname)"
echo "   commit ${git_commit}, uv.lock ${lock_sha:0:12}, out ${out_dir}, work ${work_dir}"

# 1つ作り、.partial から名前を付け替え、ハッシュと記録を残す
# 引数: 名前 定義ファイル 環境名 追加の apptainer build の引数...
build_one() {
    local name="$1" def="$2" env_name="$3"; shift 3
    local target="$out_dir/${name}.sif"
    local partial="$out_dir/.${name}.sif.partial"
    if [ -e "$target" ]; then
        die "$target already exists. Published images are never overwritten; raise project.version in pyproject.toml."
    fi
    rm -f "$partial"
    echo "== building $target"
    (cd "$ENVS_DIR" && apptainer build "$@" "$partial" "containers/$def")
    chmod 0444 "$partial"
    mv "$partial" "$target"
    (cd "$out_dir" && sha256sum "${name}.sif" > "${name}.sha256")
    python3 - "$out_dir/${name}.json" <<PY
import json, os, sys, datetime
record = {
    "name": "${name}.sif", "env": "${env_name}", "version": "${version}", "arch": "${arch}",
    "git_commit": "${git_commit}", "uv_lock_sha256": "${lock_sha}",
    "cuda_image_tag": "${cuda_tag}", "uv_version": "${uv_version}", "python_version": "${python_version}",
    "es_version": "${es_version}" if "${env_name}" == "es" else None,
    "sha256": open("$out_dir/${name}.sha256").read().split()[0],
    "size_bytes": os.path.getsize("$target"),
    "built_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "built_on": os.uname().nodename,
    "apptainer": "$(apptainer --version)",
}
json.dump(record, open(sys.argv[1], "w"), indent=2, ensure_ascii=False)
print("   recorded", sys.argv[1])
PY
}

base_name="base-${version}-${arch}"
base_sif="$out_dir/${base_name}.sif"
need_base=0
for env in ${want_envs//,/ }; do
    case "$env" in
        nlp|llm|train) need_base=1 ;;
        es) ;;
        *) die "unknown env '$env' (choose from $ALL_ENVS)" ;;
    esac
done

if [ "$need_base" -eq 1 ] && [ ! -e "$base_sif" ]; then
    build_one "$base_name" base.def base \
        --build-arg "CUDA_IMAGE_TAG=${cuda_tag}" \
        --build-arg "UV_VERSION=${uv_version}" \
        --build-arg "PYTHON_VERSION=${python_version}"
elif [ "$need_base" -eq 1 ]; then
    echo "== reusing $base_sif (same version and CPU)"
fi

for env in ${want_envs//,/ }; do
    case "$env" in
        nlp) extras="" ;;
        llm) extras="--extra llm" ;;
        train) extras="--extra train" ;;
        es)
            build_one "es-${version}-${arch}" es.def es \
                --build-arg "ES_VERSION=${es_version}" \
                --build-arg "VERSION=${version}" \
                --build-arg "GIT_COMMIT=${git_commit}"
            continue ;;
    esac
    build_one "${env}-${version}-${arch}" env.def "$env" \
        --bind "$work_dir/uv-cache:/uv-cache" \
        --build-arg "BASE_SIF=${base_sif}" \
        --build-arg "ENV_NAME=${env}" \
        --build-arg "EXTRAS=${extras}" \
        --build-arg "VERSION=${version}" \
        --build-arg "GIT_COMMIT=${git_commit}" \
        --build-arg "UV_CACHE_DIR=/uv-cache"
done

echo "== done. Next: bash envs/test.sh --out ${out_dir} on a GPU node, then record the .sha256 lines in envs/RELEASES.md"
