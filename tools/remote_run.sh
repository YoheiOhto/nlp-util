#!/usr/bin/env bash
# Stage a repository to a configured remote, submit there, and optionally collect results.
set -euo pipefail

action="${1:-plan}"
remote="${RESEARCH_REMOTE:-}"
remote_root="${RESEARCH_REMOTE_ROOT:-}"
exp="${EXP_ID:-}"
for arg in "${@:2}"; do
    case "$arg" in
        --remote=*) remote=${arg#*=} ;;
        --root=*) remote_root=${arg#*=} ;;
        --exp=*) exp=${arg#*=} ;;
        *) printf 'Unknown argument: %s\n' "$arg" >&2; exit 2 ;;
    esac
done
[ -n "$remote" ] && [ -n "$remote_root" ] || {
    printf 'Set RESEARCH_REMOTE and RESEARCH_REMOTE_ROOT (or --remote/--root).\n' >&2; exit 2;
}
local_root=$(git rev-parse --show-toplevel)
repo_name=$(basename "$local_root")
destination="${remote}:${remote_root%/}/${repo_name}/"

rsync_common=(-az --partial --human-readable --exclude .git --exclude .venv --exclude data --exclude outputs --exclude logs)
case "$action" in
    plan)
        printf 'stage:   %s -> %s\n' "$local_root/" "$destination"
        printf 'submit:  ssh %s cd %s/%s && tools/submit.sh %s\n' "$remote" "$remote_root" "$repo_name" "${exp:-latest}"
        printf 'collect: %s:%s/%s/{outputs,logs}/ -> local repository\n' "$remote" "$remote_root" "$repo_name"
        ;;
    stage)
        rsync "${rsync_common[@]}" --dry-run "$local_root/" "$destination"
        ;;
    stage-apply)
        rsync "${rsync_common[@]}" "$local_root/" "$destination"
        ;;
    submit)
        [ -n "$exp" ] || { printf '%s\n' '--exp is required for submit' >&2; exit 2; }
        ssh "$remote" "cd '$remote_root/$repo_name' && bash tools/submit.sh '$exp'"
        ;;
    collect)
        rsync -az --partial "$remote:$remote_root/$repo_name/outputs/" "$local_root/outputs/"
        rsync -az --partial "$remote:$remote_root/$repo_name/logs/" "$local_root/logs/"
        ;;
    *) printf 'Usage: remote_run.sh plan|stage|stage-apply|submit|collect [--remote=] [--root=] [--exp=]\n' >&2; exit 2 ;;
esac
