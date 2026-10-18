#!/usr/bin/env bash
# Unified Slurm/PBS/local submission entry point.
set -euo pipefail

input="${1:-}"
shift || true
allow_dirty=0
scheduler="auto"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --allow-dirty) allow_dirty=1 ;;
        --scheduler) shift; scheduler="${1:?scheduler value required}" ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

root=$(git rev-parse --show-toplevel)
if [ "$allow_dirty" -eq 0 ] && [ -n "$(git -C "$root" status --porcelain)" ]; then
    printf 'Refusing dirty submission. Commit changes or pass --allow-dirty.\n' >&2
    exit 1
fi

resolve_exp() {
    local value="$1" found
    if [[ "$value" =~ ^[0-9]+$ ]]; then value=$(printf '%04d' "$((10#$value))"); fi
    if [ -n "$value" ]; then
        found=$(find "$root/experiments" -mindepth 1 -maxdepth 1 -type d -name "${value}*" | sort)
    else
        found=$(find "$root/experiments" -mindepth 1 -maxdepth 1 -type d -name '[0-9][0-9][0-9][0-9]_*' | sort | tail -n 1)
    fi
    [ "$(printf '%s\n' "$found" | sed '/^$/d' | wc -l | tr -d ' ')" -eq 1 ] || {
        printf 'Experiment selector is missing or ambiguous: %s\n' "$value" >&2; return 1;
    }
    printf '%s\n' "$found"
}

exp_dir=$(resolve_exp "$input")
exp_name=$(basename "$exp_dir")
python3 "$root/scripts/preflight_check.py" "$exp_dir"
mkdir -p "$root/logs/$exp_name" "$root/outputs/$exp_name"

if [ "$scheduler" = auto ]; then
    if command -v sbatch >/dev/null 2>&1; then scheduler=slurm
    elif command -v qsub >/dev/null 2>&1; then scheduler=pbs
    else scheduler=local
    fi
fi

case "$scheduler" in
    slurm)
        output=$(sbatch "$exp_dir/run_slurm.sh")
        job_id=$(printf '%s\n' "$output" | grep -Eo '[0-9]+' | tail -n 1)
        ;;
    pbs)
        output=$(qsub "$exp_dir/run_qsub.sh")
        job_id=${output%%.*}
        ;;
    local)
        "$exp_dir/run_local.sh" &
        job_id=$!
        output="Local process ${job_id}"
        ;;
    *) printf 'Unsupported scheduler: %s\n' "$scheduler" >&2; exit 2 ;;
esac

printf '%s\n' "$output"
printf '%s\n' "$job_id" > "$root/outputs/$exp_name/latest_job_id.txt"
printf '{"scheduler":"%s","job_id":"%s","experiment":"%s"}\n' \
    "$scheduler" "$job_id" "$exp_name" > "$root/outputs/$exp_name/submission.json"
