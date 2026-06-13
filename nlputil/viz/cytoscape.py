from __future__ import annotations
from pathlib import Path

import pandas as pd


def edges_to_cytoscape(
    edges_df: pd.DataFrame,
    source_col: str = "node_a",
    target_col: str = "node_b",
    output_path: Path | None = None,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Convert an edge DataFrame to Cytoscape-compatible format.

    Renames source/target columns to the names expected by Cytoscape's
    CSV importer and optionally writes the result to disk.

    Args:
        edges_df: DataFrame with at least source_col and target_col.
        source_col: Column to map to "source".
        target_col: Column to map to "target".
        output_path: If provided, write CSV here.
        extra_cols: Additional columns to retain (e.g. ["weight", "pmid"]).

    Returns:
        Cytoscape-compatible DataFrame with "source" and "target" columns.
    """
    keep = [source_col, target_col] + (extra_cols or [])
    cyto = edges_df[keep].rename(columns={source_col: "source", target_col: "target"})
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cyto.to_csv(output_path, index=False)
    return cyto
