#!/usr/bin/env python3
"""Create slide-oriented daily or weekly research reports."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path


def completed_runs(root: Path, since: datetime) -> list[dict]:
    rows = []
    for path in root.glob("outputs/*/*/completion.json"):
        try:
            row = json.loads(path.read_text())
            stamp = datetime.fromisoformat(row.get("completed_at", row.get("started_at", "")))
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        if stamp >= since:
            row["experiment"] = path.parents[1].name
            row["variant"] = path.parent.name
            rows.append(row)
    return sorted(rows, key=lambda x: x.get("completed_at", ""))


def _section(path: Path, heading: str) -> str:
    if not path.exists():
        return "TODO"
    import re
    match = re.search(rf"^## {re.escape(heading)}\s*$\n(.*?)(?=^## |\Z)", path.read_text(), re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else "TODO"


def render(period: str, rows: list[dict], root: Path) -> str:
    title = f"{period.capitalize()} Research Report — {date.today().isoformat()}"
    experiments = sorted({row["experiment"] for row in rows})
    table = ["| Experiment | Variant | Status | Primary result |", "|---|---|---|---|"]
    for row in rows:
        metric = row.get("primary_metric", row.get("metric", "—"))
        table.append(f"| {row['experiment']} | {row['variant']} | {row.get('status', 'unknown')} | {metric} |")
    if not rows:
        table.append("| — | — | No completed runs | — |")
    return "\n".join([
        f"# {title}", "", "## Slide 1 — Executive summary", "",
        f"- Completed runs: {len(rows)}", f"- Experiments touched: {len(experiments)}",
        f"- Research question: {_section(root / 'RESEARCH.md', 'Research question')}",
        f"- Current objective: {_section(root / 'ROADMAP.md', 'Current objective')}",
        "- Main conclusion: TODO", "- Decision required: TODO", "",
        "## Slide 2 — Evidence", "", *table, "",
        "## Slide 3 — Interpretation and risks", "", "- What changed: TODO",
        "- Limitation / confounder: TODO", "- Failed or inconclusive runs: TODO", "",
        "## Slide 4 — Next actions", "", _section(root / "ROADMAP.md", "Next actions"), "",
        "### Decisions needed", "", _section(root / "ROADMAP.md", "Decisions needed"), "",
        "## Appendix — Reproducibility", "", "- Git commits and resource usage are available under `logs/`.", "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("period", choices=("daily", "weekly"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    days = 1 if args.period == "daily" else 7
    content = render(args.period, completed_runs(args.root, datetime.now() - timedelta(days=days)), args.root)
    output = args.output or args.root / "reports" / f"{args.period}_{date.today().isoformat()}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    print(output)


if __name__ == "__main__":
    main()
