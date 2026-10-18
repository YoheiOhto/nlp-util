#!/usr/bin/env bash
# Source this file from a user's interactive shell configuration.
# It only defines experiment-management helpers and has no global side effects.

TEMPLATE_DAILY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for template_daily_file in \
    "${TEMPLATE_DAILY_ROOT}/.bashrc.d/0-bootstrap.sh" \
    "${TEMPLATE_DAILY_ROOT}/.bashrc.d/1-experiments.sh"; do
    if [ -r "${template_daily_file}" ]; then
        # shellcheck source=/dev/null
        source "${template_daily_file}"
    else
        printf 'WARNING: template shell helper is missing: %s\n' "${template_daily_file}" >&2
    fi
done

# .bashrc.d/2-git.sh is intentionally excluded. It changes global Git hook
# configuration and is an opt-in personal helper, not template behavior.
unset template_daily_file TEMPLATE_DAILY_ROOT
