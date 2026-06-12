"""
PubMed parquet access utilities.

Efficiently reads title, abstract, and metadata from a large PubMed parquet
file using PyArrow push-down filters — only the requested PMIDs are read,
not the full dataset.
"""

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

DEFAULT_ABSTRACT_MAX_CHARS = 3000


def fetch_abstracts(
    pmids:        set[str],
    pubmed_path:  Path,
    logger:       logging.Logger,
    max_chars:    int = DEFAULT_ABSTRACT_MAX_CHARS,
) -> dict[str, dict]:
    """
    Fetch titles and abstracts for a set of PMIDs from a PubMed parquet file.

    Uses PyArrow push-down filtering so only matching rows are read from disk,
    which is critical for large parquet files (e.g. 16 GB PubMed snapshot).

    Args:
        pmids:        set of PMID strings to fetch
        pubmed_path:  path to the PubMed parquet file
        logger:       logger instance
        max_chars:    truncate abstract at this length (appends "...")

    Returns:
        {pmid_str: {"title": str, "abstract": str, "year": ..., "journal": str}}

    Notes:
        - If the parquet has no "abstract" column, returns title/year/journal only.
        - PMIDs not found in the parquet are silently omitted from the result.
    """
    logger.info(f"  Fetching abstracts for {len(pmids):,} PMIDs ...")
    valid_pmids = [p for p in pmids if str(p).lstrip("-").isdigit()]
    pmid_ints   = pa.array(sorted(int(p) for p in valid_pmids), type=pa.int64())
    dataset     = ds.dataset(str(pubmed_path), format="parquet")
    filt        = dataset.filter(pc.is_in(pc.field("pmid"), value_set=pmid_ints))

    cols         = ["pmid", "title", "year", "journal"]
    has_abstract = "abstract" in dataset.schema.names
    if has_abstract:
        cols.append("abstract")
    else:
        logger.warning("  'abstract' column not found in PubMed parquet.")

    df = filt.to_table(columns=cols).to_pandas()
    logger.info(f"  Found {len(df):,} / {len(pmids):,} PMIDs")

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        pmid     = str(int(row["pmid"]))
        abstract = ""
        if has_abstract and pd.notna(row.get("abstract")):
            abstract = str(row["abstract"])
            if len(abstract) > max_chars:
                abstract = abstract[:max_chars] + "..."
        result[pmid] = {
            "title":    str(row.get("title") or ""),
            "abstract": abstract,
            "year":     row.get("year"),
            "journal":  str(row.get("journal") or ""),
        }
    return result


def fetch_article_metadata(
    pmids:        set[int],
    parquet_path: Path,
    logger:       logging.Logger,
) -> dict[int, dict]:
    """
    Fetch title, year, and journal for a set of PMIDs (no abstract).

    Lightweight alternative to fetch_abstracts when abstract text is not needed.

    Args:
        pmids:        set of integer PMIDs
        parquet_path: path to the PubMed parquet file
        logger:       logger instance

    Returns:
        {pmid_int: {"title": str, "year": ..., "journal": str}}
    """
    logger.info(f"  Fetching metadata for {len(pmids):,} PMIDs ...")
    pmid_arr = pa.array(sorted(pmids), type=pa.int64())
    dataset  = ds.dataset(str(parquet_path), format="parquet")
    filt     = dataset.filter(pc.is_in(pc.field("pmid"), value_set=pmid_arr))
    table    = filt.to_table(columns=["pmid", "title", "year", "journal"])
    df       = table.to_pandas()
    logger.info(f"  Found {len(df):,} / {len(pmids):,} PMIDs")

    return {
        int(row["pmid"]): {
            "title":   row["title"] or "",
            "year":    row["year"],
            "journal": row["journal"] or "",
        }
        for _, row in df.iterrows()
    }
