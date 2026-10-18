#!/usr/bin/env python3
"""Atomic top-level YAML-like metadata without third-party dependencies."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


def load(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"^([A-Za-z_][\w]*):\s*(.*)$", line)
        if match:
            data[match.group(1)] = match.group(2).strip().strip("\"'")
    return data


def quote(value: str) -> str:
    if not value or re.search(r"[:#\n\r]|^\s|\s$", value):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def save(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text("".join(f"{key}: {quote(str(value))}\n" for key, value in data.items()))
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("pairs", nargs="*")
    parser.add_argument("--get")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    data = {} if args.replace else load(args.path)
    if args.get:
        print(data.get(args.get, ""))
        return
    for pair in args.pairs:
        key, separator, value = pair.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][\w]*", key):
            raise SystemExit(f"invalid key=value: {pair}")
        data[key] = value
    save(args.path, data)


if __name__ == "__main__":
    main()
