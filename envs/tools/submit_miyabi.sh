#!/usr/bin/env bash
# Submit envs/tools/build.sh to Miyabi (PBS) from a Miyabi login node.
#
# The architecture follows from the queue family and is verified again inside
# the job (build.sh stops if uname -m disagrees):
#   *-g, *-mig : Miyabi-G, aarch64, GH200 (sm_90)
#   *-c        : Miyabi-C, x86_64, no GPU (smoke checks run without CUDA)
#
# Miyabi time is a hard budget, so this script only prints the job unless
# --submit is given. Before submitting it runs miyabi_preflight.py (read-only
# queue/group/limit snapshot) when the env repository is available.
#
# Usage:
#   submit_miyabi.sh --group GROUP [options] -- <build.sh targets and options>
# Options:
#   --group NAME       PBS group_list (required, e.g. gg17)
#   --queue NAME       debug-g|short-g|regular-g|*-mig|*-c (default short-g)
#   --walltime HH:MM:SS (default 04:00:00; debug-* allows at most 00:30:00)
#   --submit           actually run qsub (default: dry-run)
# Example:
#   submit_miyabi.sh --group gg17 --queue short-g -- base profile:serve-vllm
set -euo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

GROUP=""
QUEUE="short-g"
WALLTIME="04:00:00"
SUBMIT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --group) GROUP="$2"; shift ;;
        --queue) QUEUE="$2"; shift ;;
        --walltime) WALLTIME="$2"; shift ;;
        --submit) SUBMIT=1 ;;
        -h | --help) sed -n '2,22p' "$0"; exit 0 ;;
        --) shift; break ;;
        *) nlpenv_die "unknown option $1 (build targets go after --)" ;;
    esac
    shift
done
BUILD_ARGS=("$@")
[[ -n "${GROUP}" ]] || nlpenv_die "--group is required"
[[ ${#BUILD_ARGS[@]} -gt 0 ]] || nlpenv_die "no build targets (e.g. -- base profile:serve-vllm)"

case "${QUEUE}" in
    debug-g | short-g | regular-g | debug-mig | short-mig | regular-mig) ARCH=aarch64 ;;
    debug-c | short-c | regular-c) ARCH=x86_64 ;;
    small-* | medium-* | large-* | x-large-*)
        nlpenv_die "${QUEUE} is an internal routing queue; submit to regular-g/regular-c" ;;
    *) nlpenv_die "unknown Miyabi queue ${QUEUE}" ;;
esac
if [[ "${QUEUE}" == debug-* ]]; then
    IFS=: read -r h m s <<<"${WALLTIME}"
    (( 10#${h} * 3600 + 10#${m} * 60 + 10#${s} <= 1800 )) \
        || nlpenv_die "${QUEUE} allows at most 00:30:00 (got ${WALLTIME})"
fi

LOG_DIR="$(nlpenv_store)/logs/${ARCH}"
mkdir -p "${LOG_DIR}"
SCRIPT="${LOG_DIR}/qsub-$(date +%Y%m%d-%H%M%S)-${QUEUE}.sh"
cat >"${SCRIPT}" <<EOF
#!/bin/bash
#PBS -N nlpenv-${ARCH}
#PBS -q ${QUEUE}
#PBS -l select=1
#PBS -l walltime=${WALLTIME}
#PBS -W group_list=${GROUP}
#PBS -j oe
#PBS -o ${LOG_DIR}/
#PBS -r n
set -euo pipefail
cd "${NLPENV_REPO_ROOT}"
if ! command -v apptainer >/dev/null 2>&1; then
    module load apptainer
fi
export NLPENV_EXPECT_ARCH=${ARCH}
bash "${NLPENV_TOOLS_DIR}/build.sh" $(printf '%q ' "${BUILD_ARGS[@]}")
EOF

echo "--- Miyabi ${QUEUE} (${ARCH}) group=${GROUP} select=1 walltime=${WALLTIME}"
echo "    targets: ${BUILD_ARGS[*]}"
echo "    script : ${SCRIPT}"
sed 's/^/    | /' "${SCRIPT}"
echo "    token use is at most walltime[h] x 1 node x the queue's coefficient; check show_token."

PREFLIGHT="${ENV_DIR:-}/tools/agents/miyabi_preflight.py"
if [[ -f "${PREFLIGHT}" ]]; then
    echo "--- preflight (read-only)"
    if ! python3 "${PREFLIGHT}" --queue "${QUEUE}" --group "${GROUP}" --select 1; then
        # Never submit without a readable queue/limit snapshot.
        [[ ${SUBMIT} -eq 1 ]] && nlpenv_die "preflight failed; not submitting"
        echo "--- WARNING: preflight failed (dry-run continues)"
    fi
else
    echo "--- preflight skipped: ${PREFLIGHT:-ENV_DIR unset} not found; check 'qstat --limit' and 'show_token' yourself"
fi

if [[ ${SUBMIT} -eq 1 ]]; then
    command -v qsub >/dev/null 2>&1 || nlpenv_die "qsub not found; run on a Miyabi login node"
    qsub "${SCRIPT}"
else
    echo "(dry-run) add --submit to run: qsub ${SCRIPT}"
fi
