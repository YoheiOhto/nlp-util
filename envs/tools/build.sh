#!/usr/bin/env bash
# Build nlp-util environments on the CURRENT node (run inside a Slurm/PBS job).
#
# The CPU architecture and GPU compute capability are detected on the node,
# so the same command builds x86_64 images on David and aarch64 images on
# Grace / Miyabi-G. Outputs go to envs/.store/<arch>/ (override NLPENV_STORE).
#
# Usage:
#   build.sh [options] <target>...
# Targets:
#   base              CUDA + uv base SIF (needed by every other target)
#   profile:<name>    one uv venv profile (see envs/profiles/)
#   profiles          every profile
#   llamacpp          llama.cpp CUDA SIF
#   strata            clone Strata + its venv + hardware check (no model download)
#   all               base, profiles, llamacpp, strata
# Options:
#   --relock          drop the store-side uv.lock and re-resolve (also done
#                     automatically when pyproject.toml changed)
#   --rebuild         rebuild SIF images even if they exist
#   --with-flash-attn also build the flash-attn wheel for train-hf (1-3 h)
#   --no-smoke        skip smoke checks (not recommended)
#   --require-gpu     fail smoke checks without a GPU (default: auto, required
#                     whenever nvidia-smi sees a GPU)
set -euo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RELOCK=0
REBUILD=0
WITH_FLASH=0
SMOKE=1
REQUIRE_GPU="auto"
TARGETS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --relock) RELOCK=1 ;;
        --rebuild) REBUILD=1 ;;
        --with-flash-attn) WITH_FLASH=1 ;;
        --no-smoke) SMOKE=0 ;;
        --require-gpu) REQUIRE_GPU=1 ;;
        -h | --help) sed -n '2,25p' "$0"; exit 0 ;;
        -*) nlpenv_die "unknown option $1" ;;
        *) TARGETS+=("$1") ;;
    esac
    shift
done
[[ ${#TARGETS[@]} -gt 0 ]] || nlpenv_die "no target given (try: build.sh all)"

nlpenv_require_apptainer
ARCH="$(nlpenv_detect_arch)"
if [[ -n "${NLPENV_EXPECT_ARCH:-}" && "${NLPENV_EXPECT_ARCH}" != "${ARCH}" ]]; then
    nlpenv_die "job expected ${NLPENV_EXPECT_ARCH} but $(hostname) is ${ARCH}"
fi
GPU_CAPS="$(nlpenv_detect_gpu_caps)"
CUDA_ARCH_LIST="$(nlpenv_cuda_arch_list "${ARCH}")"
STORE="$(nlpenv_store)"
ARCH_DIR="$(nlpenv_arch_dir "${ARCH}")"
BASE_SIF="$(nlpenv_sif_path "${ARCH}" base)"
LLAMA_SIF="$(nlpenv_sif_path "${ARCH}" llamacpp)"
CPUS="${SLURM_CPUS_PER_TASK:-${NCPUS:-$(nproc)}}"
if [[ "${REQUIRE_GPU}" == "auto" ]]; then
    REQUIRE_GPU=$([[ -n "${GPU_CAPS}" ]] && echo 1 || echo 0)
fi

mkdir -p "${ARCH_DIR}"/{sif,venvs,python,manifest,reports,cache,tmp} "${STORE}/logs/${ARCH}" "${STORE}/cache/uv"
LOG="${STORE}/logs/${ARCH}/build-$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-${PBS_JOBID:-local}}.log"
exec > >(tee -a "${LOG}") 2>&1

# One build per arch at a time; a second job waits instead of corrupting venvs.
exec 9>"${ARCH_DIR}/.build.lock"
flock 9

export APPTAINER_CACHEDIR="${STORE}/cache/apptainer-${ARCH}"
export APPTAINER_TMPDIR="${TMPDIR:-${ARCH_DIR}/tmp}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

GIT_COMMIT="$(git -C "${NLPENV_REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git -C "${NLPENV_REPO_ROOT}" status --porcelain -- envs 2>/dev/null | grep -c . || true)"

nlpenv_log "host=$(hostname) arch=${ARCH} gpu_caps='${GPU_CAPS:-none}' cuda_archs=${CUDA_ARCH_LIST}"
nlpenv_log "store=${STORE} cpus=${CPUS} require_gpu=${REQUIRE_GPU} commit=${GIT_COMMIT} dirty_files=${GIT_DIRTY}"
nlpenv_log "targets=${TARGETS[*]} log=${LOG}"

preflight_network() {
    local url
    for url in https://pypi.org/simple/ https://download.pytorch.org/whl/cu130/ https://github.com; do
        curl -sSf -o /dev/null --max-time 20 "${url}" \
            || nlpenv_die "no network access to ${url} from $(hostname); builds need internet"
    done
}

sha256_of() {
    sha256sum "$1" | cut -d' ' -f1
}

write_manifest() {
    # $1 name, $2.. extra "key": value JSON fragments
    local name="$1"
    shift
    local extra=""
    local kv
    for kv in "$@"; do
        extra+=", ${kv}"
    done
    cat >"${ARCH_DIR}/manifest/${name}.json" <<EOF
{"name": "${name}", "arch": "${ARCH}", "host": "$(hostname)", "built_at": "$(date -Iseconds)", "git_commit": "${GIT_COMMIT}", "git_dirty_files": ${GIT_DIRTY}, "gpu_caps": "${GPU_CAPS}", "cuda_arch_list": "${CUDA_ARCH_LIST}", "apptainer": "$(apptainer --version)", "job_id": "${SLURM_JOB_ID:-${PBS_JOBID:-}}"${extra}}
EOF
}

# apptainer build, retrying with --fakeroot where unprivileged builds are
# not allowed (e.g. some Miyabi setups). Output is gzip squashfs (the default)
# because Miyabi's Apptainer 1.3.5 cannot open zstd images.
apptainer_build() {
    local out="$1" def="$2"
    shift 2
    local tmp="${out}.tmp.$$"
    rm -f "${tmp}"
    if ! apptainer build "$@" "${tmp}" "${def}"; then
        nlpenv_log "plain build failed; retrying with --fakeroot"
        rm -f "${tmp}"
        apptainer build --fakeroot "$@" "${tmp}" "${def}" || return 1
    fi
    mv -f "${tmp}" "${out}" || return 1
}

container_exec() {
    # Run a command in the base SIF with build caches; venv/python/uv dirs
    # are passed by the caller through extra --env arguments.
    local -a binds envs fwd
    mapfile -t binds < <(nlpenv_bind_args)
    mapfile -t envs < <(nlpenv_cache_env_args "${ARCH}")
    mapfile -t fwd < <(nlpenv_forward_env_args)
    apptainer exec --nv --cleanenv "${binds[@]}" "${envs[@]}" "${fwd[@]}" \
        --env "UV_CACHE_DIR=${STORE}/cache/uv" \
        --env "UV_PYTHON_INSTALL_DIR=${ARCH_DIR}/python" \
        --env "UV_PYTHON=${NLPENV_PYTHON_DEFAULT}" \
        --env "UV_PYTHON_PREFERENCE=only-managed" \
        --env "TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST}" \
        "$@"
}

build_base() {
    if [[ -f "${BASE_SIF}" && ${REBUILD} -eq 0 ]]; then
        nlpenv_log "base: exists (${BASE_SIF}); use --rebuild to replace"
        return 0
    fi
    nlpenv_log "base: building from ${NLPENV_CUDA_IMAGE_DEFAULT}"
    apptainer_build "${BASE_SIF}" "${NLPENV_ENVS_DIR}/base/base.def" \
        --build-arg "CUDA_IMAGE=${NLPENV_CUDA_IMAGE_DEFAULT}" || return 1
    write_manifest base "\"sif\": \"${BASE_SIF}\"" "\"sha256\": \"$(sha256_of "${BASE_SIF}")\"" \
        "\"cuda_image\": \"${NLPENV_CUDA_IMAGE_DEFAULT}\""
    nlpenv_log "base: done"
}

require_base() {
    [[ -f "${BASE_SIF}" ]] || nlpenv_die "base SIF missing for ${ARCH}: run 'build.sh base' first"
}

build_profile() {
    local name="$1"
    nlpenv_require_profile "${name}"
    require_base
    local proj="${NLPENV_ENVS_DIR}/profiles/${name}"
    local venv
    venv="$(nlpenv_venv_path "${ARCH}" "${name}")"
    local -a sync_args=()
    if [[ "${name}" == "train-hf" && ${WITH_FLASH} -eq 1 ]]; then
        sync_args+=(--group flash)
    fi
    # uv sync always writes uv.lock next to pyproject.toml. Locks are not kept
    # in the repository, so sync from a per-arch copy inside the store; the
    # resolved lock stays there (git-ignored) as the record of this build.
    local work="${ARCH_DIR}/projects/${name}"
    mkdir -p "${work}"
    if [[ ${RELOCK} -eq 1 ]] || ! cmp -s "${proj}/pyproject.toml" "${work}/pyproject.toml"; then
        # A changed pyproject (or --relock) re-resolves from scratch.
        rm -f "${work}/uv.lock"
    fi
    cp -f "${proj}/pyproject.toml" "${work}/pyproject.toml"
    proj="${work}"
    nlpenv_log "profile:${name}: syncing ${venv} from ${proj} (${sync_args[*]:-default groups})"
    # MAX_JOBS bounds nvcc parallelism for source builds (~6 GiB per job).
    local mem_gib
    mem_gib=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    local max_jobs=$((mem_gib / 6))
    ((max_jobs > CPUS)) && max_jobs=${CPUS}
    ((max_jobs < 1)) && max_jobs=1
    container_exec --env "UV_PROJECT_ENVIRONMENT=${venv}" --env "MAX_JOBS=${max_jobs}" \
        --env "FLASH_ATTENTION_FORCE_BUILD=TRUE" \
        "${BASE_SIF}" bash -c "cd '${proj}' && uv sync ${sync_args[*]:-}" || return 1
    write_manifest "profile-${name}" "\"venv\": \"${venv}\"" \
        "\"uv_lock_sha256\": \"$(sha256_of "${proj}/uv.lock")\"" "\"with_flash_attn\": ${WITH_FLASH}"
    if [[ ${SMOKE} -eq 1 ]]; then
        smoke_profile "${name}" || return 1
    fi
    nlpenv_log "profile:${name}: done"
}

smoke_profile() {
    local name="$1"
    local venv report
    venv="$(nlpenv_venv_path "${ARCH}" "${name}")"
    report="${ARCH_DIR}/reports/${name}.json"
    local -a gpu_flag=()
    [[ ${REQUIRE_GPU} -eq 1 ]] && gpu_flag=(--require-gpu)
    nlpenv_log "profile:${name}: smoke check -> ${report}"
    container_exec "${BASE_SIF}" "${venv}/bin/python" -I "${NLPENV_TOOLS_DIR}/smoke_check.py" \
        --profile "${name}" --report "${report}" "${gpu_flag[@]}"
}

build_llamacpp() {
    require_base
    if [[ -f "${LLAMA_SIF}" && ${REBUILD} -eq 0 ]]; then
        nlpenv_log "llamacpp: exists (${LLAMA_SIF}); use --rebuild to replace"
    else
        local cmake_archs
        cmake_archs="$(nlpenv_cmake_archs "${CUDA_ARCH_LIST}")"
        nlpenv_log "llamacpp: building ${NLPENV_LLAMACPP_REF_DEFAULT} for CUDA archs ${cmake_archs}"
        apptainer_build "${LLAMA_SIF}" "${NLPENV_ENVS_DIR}/llamacpp/llamacpp.def" \
            --build-arg "BASE_SIF=${BASE_SIF}" \
            --build-arg "LLAMACPP_REF=${NLPENV_LLAMACPP_REF_DEFAULT}" \
            --build-arg "LLAMACPP_COMMIT=${NLPENV_LLAMACPP_COMMIT_DEFAULT}" \
            --build-arg "CUDA_ARCHS=${cmake_archs}" || return 1
        write_manifest llamacpp "\"sif\": \"${LLAMA_SIF}\"" "\"sha256\": \"$(sha256_of "${LLAMA_SIF}")\"" \
            "\"ref\": \"${NLPENV_LLAMACPP_REF_DEFAULT}\"" "\"commit\": \"${NLPENV_LLAMACPP_COMMIT_DEFAULT}\""
    fi
    if [[ ${SMOKE} -eq 1 ]]; then
        nlpenv_log "llamacpp: smoke check"
        apptainer exec --nv --cleanenv "${LLAMA_SIF}" llama-server --version || return 1
        if [[ ${REQUIRE_GPU} -eq 1 ]]; then
            # The CUDA backend must list the GPU; otherwise it silently runs on CPU.
            apptainer exec --nv --cleanenv "${LLAMA_SIF}" llama-server --list-devices 2>&1 \
                | tee "${ARCH_DIR}/reports/llamacpp-devices.txt" | grep -q CUDA \
                || { nlpenv_log "llama.cpp does not see a CUDA device"; return 1; }
        fi
    fi
    nlpenv_log "llamacpp: done"
}

build_strata() {
    require_base
    local dir="${ARCH_DIR}/strata"
    if [[ ! -d "${dir}/.git" ]]; then
        git clone --depth 1 --branch "${NLPENV_STRATA_REF_DEFAULT}" https://github.com/Niko1221/Strata "${dir}" || return 1
    fi
    local actual
    actual="$(git -C "${dir}" describe --tags --exact-match 2>/dev/null || echo none)"
    [[ "${actual}" == "${NLPENV_STRATA_REF_DEFAULT}" ]] \
        || nlpenv_die "strata checkout is at '${actual}', expected ${NLPENV_STRATA_REF_DEFAULT}"
    nlpenv_log "strata: ${NLPENV_STRATA_REF_DEFAULT} hardware check (no model download)"
    # STRATA_UV=1 makes setup.sh create its venv with uv instead of sudo apt.
    container_exec --env STRATA_UV=1 "${BASE_SIF}" bash -c "cd '${dir}' && ./setup.sh --check" \
        | tee "${ARCH_DIR}/reports/strata-check.txt" || return 1
    write_manifest strata "\"dir\": \"${dir}\"" "\"ref\": \"${NLPENV_STRATA_REF_DEFAULT}\"" \
        "\"commit\": \"$(git -C "${dir}" rev-parse HEAD)\""
    nlpenv_log "strata: done (install a model with: nlpenv strata setup --model <Q> --context <N>)"
}

preflight_network
FAILED=()
run_target() {
    local t="$1"
    case "${t}" in
        base) build_base ;;
        llamacpp) build_llamacpp ;;
        strata) build_strata ;;
        profile:*) build_profile "${t#profile:}" ;;
        profiles)
            local p
            for p in "${NLPENV_PROFILES[@]}"; do
                # Keep going so one broken profile does not hide the others.
                ( build_profile "${p}" ) || FAILED+=("profile:${p}")
            done
            ;;
        all)
            build_base
            run_target profiles
            ( build_llamacpp ) || FAILED+=(llamacpp)
            ( build_strata ) || FAILED+=(strata)
            ;;
        *) nlpenv_die "unknown target '${t}'" ;;
    esac
}
for t in "${TARGETS[@]}"; do
    run_target "${t}"
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    nlpenv_log "FAILED: ${FAILED[*]} (see ${LOG})"
    exit 1
fi
nlpenv_log "all requested targets finished on ${ARCH}"
