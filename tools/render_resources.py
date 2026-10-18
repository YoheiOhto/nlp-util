#!/usr/bin/env python3
"""Render scheduler resource directives from experiment.json."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def scale(time_value: str) -> str:
    hours, minutes, seconds = (int(x) for x in time_value.split(":"))
    total = hours * 60 + minutes + (1 if seconds else 0)
    return "small" if total <= 60 else "medium" if total <= 120 else "large" if total <= 240 else "x-large"


def replace_line(text: str, prefix: str, value: str) -> str:
    result, count = re.subn(rf"^{re.escape(prefix)}.*$", prefix + value, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"directive not found: {prefix}")
    return result


def render(exp: Path, owner: str | None = None) -> None:
    resources = json.loads((exp / "experiment.json").read_text())["resources"]
    time_value = resources["time"]
    margin = max(30, int(sum(x * m for x, m in zip((int(v) for v in time_value.split(":")), (3600, 60, 1))) / 100))

    slurm = exp / "run_slurm.sh"
    text = slurm.read_text()
    current_partition = re.search(r"^#SBATCH --partition=(.*)$", text, re.MULTILINE)
    suffix = owner or (current_partition.group(1).split("-", 1)[-1] if current_partition else "david01")
    for prefix, value in (
        ("#SBATCH --partition=", f"{scale(time_value)}-{suffix}"),
        ("#SBATCH --gres=gpu:", str(resources["gpus"])),
        ("#SBATCH --cpus-per-task=", str(resources["cpus"])),
        ("#SBATCH --mem=", f"{resources['memory_gb']}g"),
        ("#SBATCH --time=", time_value),
        ("#SBATCH --signal=B:USR1@", str(margin)),
    ):
        text = replace_line(text, prefix, value)
    slurm.write_text(text)

    pbs = exp / "run_qsub.sh"
    text = pbs.read_text()
    text = replace_line(text, "#PBS -l walltime=", time_value)
    select = f"1:ncpus={resources['cpus']}:mem={resources['memory_gb']}gb"
    if int(resources["gpus"]):
        select += f":ngpus={resources['gpus']}"
    text = replace_line(text, "#PBS -l select=", select)
    pbs.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=Path)
    parser.add_argument("--owner")
    args = parser.parse_args()
    exp = args.exp
    if not exp.is_dir():
        selector = str(exp).zfill(4) if str(exp).isdigit() else str(exp)
        matches = sorted(Path("experiments").glob(f"{selector}*"))
        if len(matches) != 1:
            raise SystemExit(f"expected one experiment for {args.exp}, found {len(matches)}")
        exp = matches[0]
    render(exp, args.owner)
    print(f"rendered resources: {exp}")


if __name__ == "__main__":
    main()
