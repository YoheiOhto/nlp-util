from __future__ import annotations
import json
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return a list of records."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    """Write records to a JSONL file, overwriting if it exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(record: dict, path: Path) -> None:
    """Append a single record to a JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_parquet_filtered(
    path: Path,
    column: str,
    values: list,
) -> pd.DataFrame:
    """Read parquet with PyArrow push-down filter on a single column.

    Much faster than reading the full file when the value set is small.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path, filters=[(column, "in", set(values))])
    return table.to_pandas()
