#!/usr/bin/env bash
# Submit envs/tools/build.sh as Slurm jobs, choosing partitions automatically.
#
# How the target is chosen:
#   1. The storage zone of this checkout decides which node families can see it
#      (/workspace/david01 is local to david01; filesrv01/02 are shared with the
#      creator and grace families).
#   2. For each requested CPU architecture, the family's real architecture is
#      read from `scontrol show node` (never from a hard-coded table), and a
#      mismatch stops the submission.
#   3. The size class (small/medium/large/x-large) is the smallest whose
#      MaxTime fits --time; CPU and memory default to the whole node.
# The build itself detects the architecture again on the compute node.
#
# Usage:
#   submit_slurm.sh [options] -- <build.sh targets and options>
# Options:
#   --arch auto|x86_64|aarch64  default auto = every arch reachable from this checkout
#   --family NAME              force a node family (e.g. david01, grace-o, preproc-o)
#   --time HH:MM:SS            walltime (default 04:00:00)
#   --cpus N / --mem MB        override the whole-node defaults
#   --dry-run                  print the job script and sbatch command only
# Examples:
#   submit_slurm.sh -- all
#   submit_slurm.sh --arch aarch64 -- base profile:serve-vllm
#   submit_slurm.sh --time 08:00:00 -- --with-flash-attn profile:train-hf
set -euo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ARCH_REQ="auto"
FAMILY=""
TIME="04:00:00"
CPUS=""
MEM=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch) ARCH_REQ="$2"; shift ;;
        --family) FAMILY="$2"; shift ;;
        --time) TIME="$2"; shift ;;
        --cpus) CPUS="$2"; shift ;;
        --mem) MEM="$2"; shift ;;
        --dry-run) DRY_RUN=1 ;;
        -h | --help) sed -n '2,29p' "$0"; exit 0 ;;
        --) shift; break ;;
        *) nlpenv_die "unknown option $1 (build targets go after --)" ;;
    esac
    shift
done
BUILD_ARGS=("$@")
[[ ${#BUILD_ARGS[@]} -gt 0 ]] || nlpenv_die "no build targets (e.g. submit_slurm.sh -- all)"
command -v sbatch >/dev/null 2>&1 || nlpenv_die "sbatch not found; is this a Slurm login node?"

REPO_REAL="$(readlink -f "${NLPENV_REPO_ROOT}")"

# Node families that can see this checkout, per arch.
families_for_zone() {
    local arch="$1"
    case "${REPO_REAL}" in
        /workspace/david01/*) [[ "${arch}" == x86_64 ]] && echo david01 ;;
        /workspace/david02/*) [[ "${arch}" == x86_64 ]] && echo david02 ;;
        /workspace/andre01/*) [[ "${arch}" == x86_64 ]] && echo andre01 ;;
        /workspace/filesrv01/*) [[ "${arch}" == x86_64 ]] && echo creator-o || echo grace-o ;;
        /workspace/filesrv02/*) [[ "${arch}" == x86_64 ]] && echo creator-i || echo grace-i ;;
        *) nlpenv_die "cannot map ${REPO_REAL} to a Slurm storage zone; pass --family" ;;
    esac
    return 0
}

# Real CPU architecture of a family, from the nodes of its small-* partition.
family_arch() {
    local family="$1" part nodes arches
    part="$(partition_for "${family}" "00:01:00")"
    nodes="$(sinfo -h -p "${part}" -o '%N')"
    [[ -n "${nodes}" ]] || nlpenv_die "partition ${part} has no nodes"
    # DOWN/NOT_RESPONDING nodes report no Arch; only responding nodes count.
    arches="$(scontrol show node "${nodes}" -o | grep -oE 'Arch=[^ ]+' | sed 's/Arch=//' | grep . | sort -u || true)"
    [[ -n "${arches}" ]] || nlpenv_die "no responding node in ${part} reports its Arch (nodes: ${nodes})"
    [[ "$(echo "${arches}" | wc -l)" -eq 1 ]] || nlpenv_die "${family} mixes architectures: ${arches}"
    echo "${arches}"
}

to_seconds() {
    local t="$1" d=0 h m s
    if [[ "${t}" == *-* ]]; then
        d="${t%%-*}"
        t="${t#*-}"
    fi
    IFS=: read -r h m s <<<"${t}"
    echo $((d * 86400 + 10#${h} * 3600 + 10#${m} * 60 + 10#${s:-0}))
}

# Smallest size class of a family whose MaxTime fits the request.
partition_for() {
    local family="$1" want cls limit
    want="$(to_seconds "$2")"
    for cls in small medium large x-large; do
        limit="$(sinfo -h -p "${cls}-${family}" -o '%l' 2>/dev/null | head -1)"
        [[ -n "${limit}" ]] || continue
        if [[ "${limit}" == infinite ]] || ((want <= $(to_seconds "${limit}"))); then
            echo "${cls}-${family}"
            return 0
        fi
    done
    nlpenv_die "no ${family} partition accepts --time ${2}"
}

submit_one() {
    local family="$1" arch="$2"
    local part cpus mem log_dir script
    part="$(partition_for "${family}" "${TIME}")"
    cpus="${CPUS:-$(sinfo -h -p "${part}" -o '%c' | head -1 | tr -d '+')}"
    # Leave 10% of node memory for the OS and page cache.
    mem="${MEM:-$(($(sinfo -h -p "${part}" -o '%m' | head -1 | tr -d '+') * 9 / 10))}"
    log_dir="$(nlpenv_store)/logs/${arch}"
    mkdir -p "${log_dir}"
    script="${log_dir}/submit-$(date +%Y%m%d-%H%M%S)-${family}.sh"
    cat >"${script}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=nlpenv-${arch}
#SBATCH --partition=${part}
#SBATCH --time=${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}M
#SBATCH --gres=gpu:1
#SBATCH --chdir=${NLPENV_REPO_ROOT}
#SBATCH --output=${log_dir}/slurm-%j.out
set -euo pipefail
export NLPENV_EXPECT_ARCH=${arch}
bash "${NLPENV_TOOLS_DIR}/build.sh" $(printf '%q ' "${BUILD_ARGS[@]}")
EOF
    echo "--- ${arch} on ${part}: cpus=${cpus} mem=${mem}M time=${TIME} gpu=1"
    echo "    targets: ${BUILD_ARGS[*]}"
    echo "    script : ${script}"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        sed 's/^/    | /' "${script}"
        echo "    (dry-run) sbatch ${script}"
    else
        sbatch "${script}"
    fi
}

ARCHES=()
case "${ARCH_REQ}" in
    auto) ARCHES=(x86_64 aarch64) ;;
    x86_64 | aarch64) ARCHES=("${ARCH_REQ}") ;;
    *) nlpenv_die "--arch must be auto, x86_64 or aarch64" ;;
esac

SUBMITTED=0
if [[ -n "${FAMILY}" ]]; then
    submit_one "${FAMILY}" "$(family_arch "${FAMILY}")"
    SUBMITTED=1
else
    for arch in "${ARCHES[@]}"; do
        family="$(families_for_zone "${arch}")"
        if [[ -z "${family}" ]]; then
            [[ "${ARCH_REQ}" == auto ]] && { echo "--- ${arch}: no node family can see ${REPO_REAL}; skipped"; continue; }
            nlpenv_die "${arch}: no node family can see ${REPO_REAL} (clone the repo on filesrv01/02 for aarch64)"
        fi
        real="$(family_arch "${family}")"
        [[ "${real}" == "${arch}" ]] || nlpenv_die "${family} reports Arch=${real}, expected ${arch}"
        submit_one "${family}" "${arch}"
        SUBMITTED=$((SUBMITTED + 1))
    done
fi
((SUBMITTED > 0)) || nlpenv_die "nothing submitted"
