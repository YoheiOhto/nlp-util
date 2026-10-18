#!/usr/bin/env python3
"""Estimate resources from a manifest and completed job metadata."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def _hours(value: str) -> float:
    parts = [int(x) for x in value.split(":" )]
    if len(parts) == 3:
        return parts[0] + parts[1] / 60 + parts[2] / 3600
    raise ValueError(f"invalid time: {value}")


def _read_simple_yaml(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"^([A-Za-z_][\w]*):\s*[\"']?(.*?)[\"']?\s*$", line)
        if match:
            result[match.group(1)] = match.group(2)
    return result


def estimate(root: Path, exp: Path) -> dict:
    manifest = json.loads((exp / "experiment.json").read_text())
    requested = manifest["resources"]
    history = []
    for path in (root / "logs").glob("*/**/run_metadata.yaml"):
        row = _read_simple_yaml(path)
        if row.get("status", "").upper() == "COMPLETED":
            history.append(row)

    elapsed = []
    peak_mem = []
    for row in history:
        try:
            elapsed.append(float(row["elapsed_seconds"]) / 3600)
        except (KeyError, ValueError):
            pass
        raw_mem = row.get("max_rss_gb", row.get("max_rss", ""))
        match = re.fullmatch(r"([0-9.]+)([KMGTP]?)", raw_mem, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2).upper()
            scale = {"": 1, "K": 1 / 1024**2, "M": 1 / 1024, "G": 1, "T": 1024, "P": 1024**2}
            peak_mem.append(value * scale[unit])

    time_hours = max(elapsed) * 1.25 if elapsed else _hours(requested["time"])
    mem_gb = math.ceil(max(peak_mem) * 1.2) if peak_mem else int(requested["memory_gb"])
    confidence = "medium" if len(elapsed) >= 3 else "low"
    return {
        "experiment": exp.name,
        "recommended": {
            **requested,
            "memory_gb": mem_gb,
            "time_hours": round(time_hours, 2),
        },
        "evidence": {"completed_jobs": len(history), "timed_jobs": len(elapsed)},
        "confidence": confidence,
        "note": "Low confidence means the manifest values are retained; update after measured runs.",
    }


def resolve_exp(root: Path, value: str | None) -> Path:
    candidates = sorted(p for p in (root / "experiments").glob("[0-9][0-9][0-9][0-9]_*") if p.is_dir())
    if value:
        padded = value.zfill(4) if value.isdigit() else value
        candidates = [p for p in candidates if p.name.startswith(padded) or padded in p.name]
    if len(candidates) != 1:
        raise SystemExit(f"expected one experiment, found {len(candidates)}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--exp")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = estimate(args.root.resolve(), resolve_exp(args.root.resolve(), args.exp))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        rec = result["recommended"]
        print(f"experiment: {result['experiment']}")
        print(f"GPU: {rec['gpus']} (target VRAM {rec['gpu_memory_gb']} GB)")
        print(f"CPU: {rec['cpus']}; memory: {rec['memory_gb']} GB; time: {rec['time_hours']} h")
        print(f"confidence: {result['confidence']} ({result['evidence']['timed_jobs']} measured jobs)")


if __name__ == "__main__":
    main()
