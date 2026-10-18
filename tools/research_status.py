#!/usr/bin/env python3
"""Render the research objective, roadmap, and experiment/job state."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def section(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    return match.group(1).strip() if match else "(not recorded)"


def metadata_status(path: Path) -> str:
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("status:"):
            return line.partition(":")[2].strip().strip("\"'")
    return "UNKNOWN"


def job_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in (root / "logs").glob("*/**/run_metadata.yaml"):
        status = metadata_status(path).upper()
        counts[status] = counts.get(status, 0) + 1
    return counts


def experiment_counts(root: Path) -> dict[str, int]:
    counts = {"total": 0, "completed_variants": 0, "failed_variants": 0}
    counts["total"] = sum(1 for path in (root / "experiments").glob("[0-9][0-9][0-9][0-9]_*") if path.is_dir())
    for path in (root / "outputs").glob("*/*/completion.json"):
        try:
            status = json.loads(path.read_text()).get("status", "").lower()
        except (OSError, json.JSONDecodeError):
            continue
        if status == "completed":
            counts["completed_variants"] += 1
        elif status in {"failed", "cancelled"}:
            counts["failed_variants"] += 1
    return counts


def render(root: Path) -> str:
    research = (root / "RESEARCH.md").read_text()
    roadmap = (root / "ROADMAP.md").read_text()
    experiments = experiment_counts(root)
    jobs = job_counts(root)
    jobs_text = ", ".join(f"{key}={value}" for key, value in sorted(jobs.items())) or "none"
    return "\n".join(
        [
            "RESEARCH STATUS",
            "===============",
            "",
            "Research question",
            section(research, "Research question"),
            "",
            "Current objective",
            section(roadmap, "Current objective"),
            "",
            "Next actions",
            section(roadmap, "Next actions"),
            "",
            "Blockers",
            section(roadmap, "Blockers"),
            "",
            "Decisions needed",
            section(roadmap, "Decisions needed"),
            "",
            f"Experiments: total={experiments['total']}, completed variants={experiments['completed_variants']}, failed variants={experiments['failed_variants']}",
            f"Jobs: {jobs_text}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(render(args.root.resolve()))


if __name__ == "__main__":
    main()
