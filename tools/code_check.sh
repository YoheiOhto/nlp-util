#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

# Check only versioned code. Runtime outputs and local environments can be large.
while IFS= read -r -d '' file; do
    bash -n "$file"
done < <(git ls-files -z -- '*.sh')

python3 - <<'PY'
import ast
import subprocess
from pathlib import Path

paths = subprocess.check_output(["git", "ls-files", "-z", "--", "*.py"]).split(b"\0")
for raw_path in paths:
    if raw_path:
        path = Path(raw_path.decode())
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
PY

python3 -m unittest discover -s tests -v
