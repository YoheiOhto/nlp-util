#!/usr/bin/env bash
# Shared helpers for the nlp-util environment toolkit.
#
# Sourced by build.sh, submit_slurm.sh, submit_miyabi.sh and bin/nlpenv.
# Every helper fails fast: an unknown architecture, a missing SIF or an
# unsupported GPU stops the caller instead of silently falling back.

# Resolve directories relative to this file so every checkout (David, Grace,
# Miyabi) works without editing paths.
NLPENV_TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NLPENV_ENVS_DIR="$(cd "${NLPENV_TOOLS_DIR}/.." && pwd)"
NLPENV_REPO_ROOT="$(cd "${NLPENV_ENVS_DIR}/.." && pwd)"

# Pinned upstream identities. Bump these deliberately and rebuild.
NLPENV_CUDA_IMAGE_DEFAULT="nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04"
NLPENV_PYTHON_DEFAULT="3.12"
NLPENV_LLAMACPP_REF_DEFAULT="v0.7.0"
NLPENV_LLAMACPP_COMMIT_DEFAULT="620d2225cc436ab0b001e3200e590e3b286ef9ab"
NLPENV_STRATA_REF_DEFAULT="v0.1.45"

# Profiles that are uv projects under envs/profiles/<name>.
NLPENV_PROFILES=(base bio-scispacy serve-vllm serve-vllm-0.26 serve-sglang train-hf train-unsloth allinone)

# Profiles whose "flash" dependency group is the source-built flash-attn wheel
# (installed only with build.sh --with-flash-attn).
NLPENV_FLASH_PROFILES=(train-hf allinone)

nlpenv_die() {
    echo "[nlpenv] ERROR: $*" >&2
    exit 1
}

nlpenv_log() {
    echo "[nlpenv] $(date '+%Y-%m-%dT%H:%M:%S%z') $*" >&2
}

# Print the normalized CPU architecture (x86_64 or aarch64).
nlpenv_detect_arch() {
    local machine
    machine="$(uname -m)"
    case "${machine}" in
        x86_64 | amd64) echo "x86_64" ;;
        aarch64 | arm64) echo "aarch64" ;;
        *) nlpenv_die "unsupported CPU architecture: ${machine}" ;;
    esac
}

# Print the unique GPU compute capabilities on this node, e.g. "12.0" or
# "9.0 12.0". Prints nothing when no NVIDIA GPU is visible, or when
# NLPENV_NO_GPU=1 (a CPU-only job on a GPU node, where nvidia-smi may still
# list the node's GPU even though the job was not allocated one).
nlpenv_detect_gpu_caps() {
    [[ "${NLPENV_NO_GPU:-0}" == 1 ]] && return 0
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | tr -d ' ' | grep -E '^[0-9]+\.[0-9]+$' | sort -u | tr '\n' ' ' | sed 's/ $//'
}

# Print the CUDA architectures to compile for, in TORCH_CUDA_ARCH_LIST form
# ("8.9;9.0;12.0"). The per-arch baseline keeps one image usable on every
# node of that CPU family; the detected GPU is always added. Override the
# whole list with NLPENV_CUDA_ARCHS.
#   x86_64 : Ampere/Ada (andre01, creator*), Hopper, Blackwell RTX PRO (sm_120)
#   aarch64: GH200 on Miyabi (sm_90), GB10 on Grace (sm_121 runs sm_120 SASS)
nlpenv_cuda_arch_list() {
    local arch="$1"
    if [[ -n "${NLPENV_CUDA_ARCHS:-}" ]]; then
        echo "${NLPENV_CUDA_ARCHS}"
        return 0
    fi
    local baseline
    case "${arch}" in
        x86_64) baseline="8.0 8.6 8.9 9.0 12.0" ;;
        aarch64) baseline="9.0 12.0" ;;
        *) nlpenv_die "no CUDA arch baseline for ${arch}" ;;
    esac
    local caps
    caps="$(nlpenv_detect_gpu_caps)"
    # shellcheck disable=SC2086
    printf '%s\n' ${baseline} ${caps} | sort -u -t. -k1,1n -k2,2n | tr '\n' ';' | sed 's/;$//'
}

# Convert "8.9;9.0;12.0" to CMake form "89;90;120".
nlpenv_cmake_archs() {
    echo "$1" | tr ';' '\n' | sed 's/\.//' | tr '\n' ';' | sed 's/;$//'
}

# Convert "8.0;8.6;8.9;9.0;12.0" to flash-attn 2.8's FLASH_ATTN_CUDA_ARCHS
# form "80;90;120". flash-attn's setup.py ignores TORCH_CUDA_ARCH_LIST (it
# passes its own -gencode flags) and only knows the families 80, 90, 100 and
# 120; a family's SASS runs on every minor of the same major (sm_86/sm_89 run
# sm_80 code, sm_121 runs sm_120 code).
nlpenv_flash_attn_archs() {
    local cap major out=""
    local -a seen=()
    for cap in $(echo "$1" | tr ';' ' '); do
        major="${cap%%.*}"
        case "${major}" in
            8 | 9 | 10 | 12) ;;
            *) nlpenv_die "flash-attn 2.8 has no kernels for compute capability ${cap}" ;;
        esac
        [[ " ${seen[*]} " == *" ${major}0 "* ]] && continue
        seen+=("${major}0")
        out+="${out:+;}${major}0"
    done
    [[ -n "${out}" ]] || nlpenv_die "empty CUDA arch list for flash-attn"
    echo "${out}"
}

# Print "MAX_JOBS NVCC_THREADS" for a CUDA source build.
#   $1 CPUs of the allocation, $2 usable memory in MiB, $3 peak GiB per job.
# The memory budget keeps 4 GiB for the rest of the build; nvcc threads per
# job fill the CPUs (capped at 4, flash-attn's default) so that
# MAX_JOBS * NVCC_THREADS <= CPUs.
nlpenv_compile_jobs() {
    local cpus="$1" mem_mib="$2" per_job_gib="$3"
    [[ "${cpus}" =~ ^[0-9]+$ && "${mem_mib}" =~ ^[0-9]+$ && "${per_job_gib}" =~ ^[0-9]+$ ]] \
        || nlpenv_die "nlpenv_compile_jobs needs integers, got '$*'"
    ((cpus > 0 && per_job_gib > 0)) || nlpenv_die "nlpenv_compile_jobs: cpus and per-job GiB must be > 0"
    local jobs=$(((mem_mib / 1024 - 4) / per_job_gib))
    ((jobs > cpus)) && jobs=${cpus}
    ((jobs < 1)) && jobs=1
    local threads=$((cpus / jobs))
    ((threads > 4)) && threads=4
    ((threads < 1)) && threads=1
    echo "${jobs} ${threads}"
}

# Usable memory of this job in MiB: the smaller of MemAvailable and the
# Slurm allocation (SLURM_MEM_PER_NODE). MemAvailable alone is node-wide and
# overestimates what a cgroup-limited job may use.
nlpenv_job_mem_mib() {
    local avail
    avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    if [[ -n "${SLURM_MEM_PER_NODE:-}" && "${SLURM_MEM_PER_NODE}" =~ ^[0-9]+$ ]] \
        && ((SLURM_MEM_PER_NODE < avail)); then
        avail=${SLURM_MEM_PER_NODE}
    fi
    echo "${avail}"
}

nlpenv_is_flash_profile() {
    local p
    for p in "${NLPENV_FLASH_PROFILES[@]}"; do
        [[ "${p}" == "$1" ]] && return 0
    done
    return 1
}

# Root of all built artifacts. Defaults to envs/.store inside this checkout
# (git-ignored); override with NLPENV_STORE.
nlpenv_store() {
    echo "${NLPENV_STORE:-${NLPENV_ENVS_DIR}/.store}"
}

nlpenv_arch_dir() {
    echo "$(nlpenv_store)/$1"
}

nlpenv_sif_path() {
    # $1 arch, $2 image name (base | llamacpp)
    echo "$(nlpenv_arch_dir "$1")/sif/$2.sif"
}

nlpenv_venv_path() {
    # $1 arch, $2 profile
    echo "$(nlpenv_arch_dir "$1")/venvs/$2"
}

# Hugging Face cache: HF_HOME if set, else the standard ~/.cache/huggingface,
# so models already downloaded by other projects are reused.
nlpenv_hf_home() {
    echo "${HF_HOME:-${HOME}/.cache/huggingface}"
}

nlpenv_require_profile() {
    local name="$1" p
    for p in "${NLPENV_PROFILES[@]}"; do
        [[ "${p}" == "${name}" ]] && return 0
    done
    nlpenv_die "unknown profile '${name}'. Known: ${NLPENV_PROFILES[*]}"
}

nlpenv_require_apptainer() {
    command -v apptainer >/dev/null 2>&1 || nlpenv_die "apptainer not found on $(hostname)"
}

# Environment variables forwarded into --cleanenv containers.
nlpenv_forward_env_args() {
    local var
    local -a args=()
    for var in CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES HF_TOKEN HF_HUB_OFFLINE \
        HF_HUB_ENABLE_HF_TRANSFER WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY WANDB_MODE \
        OMP_NUM_THREADS MKL_NUM_THREADS TERM TZ \
        SLURM_JOB_ID SLURM_ARRAY_TASK_ID SLURM_CPUS_PER_TASK SLURM_NTASKS PBS_JOBID; do
        if [[ -n "${!var:-}" ]]; then
            args+=(--env "${var}=${!var}")
        fi
    done
    # An empty printf would emit one blank line, i.e. an empty argv element.
    [[ ${#args[@]} -gt 0 ]] && printf '%s\n' "${args[@]}"
    return 0
}

# Container cache locations so Triton/vLLM/torch compile caches stay per arch
# and never mix x86_64 and aarch64 artifacts in $HOME.
nlpenv_cache_env_args() {
    local arch="$1"
    local cache
    cache="$(nlpenv_arch_dir "${arch}")/cache"
    printf '%s\n' \
        --env "HF_HOME=$(nlpenv_hf_home)" \
        --env "XDG_CACHE_HOME=${cache}/xdg" \
        --env "TRITON_CACHE_DIR=${cache}/triton" \
        --env "TORCHINDUCTOR_CACHE_DIR=${cache}/inductor" \
        --env "VLLM_CACHE_ROOT=${cache}/vllm" \
        --env "FLASHINFER_WORKSPACE_BASE=${cache}/flashinfer" \
        --env "CUDA_CACHE_PATH=${cache}/nv"
}

# Bind mounts: this checkout plus the HF cache and any extra paths.
nlpenv_bind_args() {
    local hf
    hf="$(nlpenv_hf_home)"
    mkdir -p "${hf}"
    printf '%s\n' --bind "${NLPENV_REPO_ROOT}" --bind "${hf}"
    local store
    store="$(nlpenv_store)"
    case "${store}" in
        "${NLPENV_REPO_ROOT}"/*) ;;
        *) printf '%s\n' --bind "${store}" ;;
    esac
}
