"""
Elasticsearch utilities for PubMed full-text search.

Key design decisions:
  - INDEX_MAPPING includes a case-sensitive analyzer for title.exact / abstract.exact
    so that gene symbols like "TP53" are not lowercased to match unrelated documents.
  - search_node_with_terms uses search_after to retrieve all results, bypassing
    the 10,000-hit limit of standard from/size pagination.
"""

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


# =============================================================================
# Index definition
# =============================================================================

_INDEX_SETTINGS = {
    "analysis": {
        "analyzer": {
            "english_stemmed": {
                "tokenizer": "standard",
                "filter":    ["lowercase", "porter_stem"],
            },
            # No lowercasing — preserves case for gene/protein symbol searches
            "case_sensitive": {
                "tokenizer": "standard",
            },
        }
    }
}

INDEX_MAPPING = {
    "settings": _INDEX_SETTINGS,
    "mappings": {
        "properties": {
            "pmid":               {"type": "long"},
            "title": {
                "type":    "text",
                "analyzer": "english_stemmed",
                "fields":  {"exact": {"type": "text", "analyzer": "case_sensitive"}},
            },
            "abstract": {
                "type":    "text",
                "analyzer": "english_stemmed",
                "fields":  {"exact": {"type": "text", "analyzer": "case_sensitive"}},
            },
            "journal":            {"type": "keyword"},
            "language":           {"type": "keyword"},
            "year":               {"type": "integer"},
            "mesh":               {"type": "keyword"},
            "publication_types":  {"type": "keyword"},
            "abstract_truncated": {"type": "integer"},
        }
    },
}


# =============================================================================
# Connection
# =============================================================================

def make_es(host: str, user: str = "", password: str = ""):
    """
    Create an Elasticsearch client.

    Args:
        host    : e.g. "http://localhost:9200"
        user    : basic auth username (empty string = no auth)
        password: basic auth password
    """
    from elasticsearch import Elasticsearch
    auth = (user, password) if user else None
    return Elasticsearch(host, basic_auth=auth, request_timeout=60)


# =============================================================================
# Index construction
# =============================================================================

def _article_actions(parquet_path: Path, index: str):
    """Yield ES bulk actions from a PubMed parquet file. Skips rows with no abstract."""
    pf = pq.ParquetFile(str(parquet_path))
    for batch in pf.iter_batches(batch_size=50_000):
        d = batch.to_pydict()
        abstracts = d.get("abstract", [None] * len(d["pmid"]))
        for i in range(len(d["pmid"])):
            abstract = abstracts[i] or ""
            if not abstract:
                continue
            yield {
                "_index": index,
                "_id":    d["pmid"][i],
                "_source": {
                    "pmid":               d["pmid"][i],
                    "title":              (d["title"][i] or ""),
                    "abstract":           abstract,
                    "journal":            (d["journal"][i] or "") if "journal" in d else "",
                    "language":           (d["language"][i] or "") if "language" in d else "",
                    "year":               d["year"][i] if "year" in d else None,
                    "mesh":               (d["mesh"][i] or []) if "mesh" in d else [],
                    "publication_types":  (d["publication_types"][i] or []) if "publication_types" in d else [],
                    "abstract_truncated": d["abstract_truncated"][i] if "abstract_truncated" in d else 0,
                },
            }


def ensure_index(
    es,
    index:         str,
    parquet_path:  Path,
    logger:        logging.Logger,
    force_rebuild: bool = False,
) -> None:
    """
    Build an ES index from a PubMed parquet file if it does not already exist.

    Idempotent: skips building if the index exists and force_rebuild is False.
    Uses parallel_bulk with 16 threads for throughput on large parquet files.

    Args:
        es:            Elasticsearch client
        index:         target index name
        parquet_path:  PubMed parquet file
        logger:        logger instance
        force_rebuild: if True, delete and rebuild even if index exists
    """
    from elasticsearch import helpers

    if es.indices.exists(index=index):
        if force_rebuild:
            logger.info(f"Index '{index}' exists. Rebuilding (force_rebuild=True) ...")
            es.indices.delete(index=index)
        else:
            logger.info(f"Index '{index}' already exists. Skipping build.")
            return

    pf_meta   = pq.ParquetFile(str(parquet_path))
    total_raw = pf_meta.metadata.num_rows
    logger.info(f"Building index '{index}' from {parquet_path} ({total_raw:,} rows) ...")
    es.indices.create(index=index, body=INDEX_MAPPING)

    success, failed = 0, 0
    with tqdm(total=total_raw, unit="doc", unit_scale=True,
              desc="Indexing", dynamic_ncols=True) as pbar:
        for ok, _ in helpers.parallel_bulk(
            es,
            _article_actions(parquet_path, index),
            chunk_size=20_000,
            thread_count=16,
            queue_size=16,
            raise_on_error=False,
        ):
            if ok:
                success += 1
            else:
                failed += 1
            pbar.update(1)

    es.indices.refresh(index=index)
    count = es.count(index=index)["count"]
    logger.info(f"  Indexed: {success:,}  Failed: {failed:,}  Total in ES: {count:,}")


# =============================================================================
# Search
# =============================================================================

def search_node_with_terms(
    es,
    index:            str,
    terms:            list[str],
    batch_size:       int = 1000,
    include_abstract: bool = False,
) -> list[dict]:
    """
    Phrase-search for any of the given terms in title.exact and abstract.exact.

    Uses OR (bool/should) across terms and search_after pagination to retrieve
    all matching documents, bypassing the ES 10,000-hit limit.

    When the same PMID matches multiple terms, only the highest-scoring hit
    is returned.

    Args:
        es:               Elasticsearch client
        index:            index name
        terms:            list of search terms (phrase search, case-sensitive)
        batch_size:       hits per page (default 1000)
        include_abstract: if True, include abstract text in results

    Returns:
        list of {pmid, title, year, journal, es_score[, abstract]}
    """
    clean_terms = [t for t in terms if t and isinstance(t, str)]
    if not clean_terms:
        return []

    source_fields = ["pmid", "title", "year", "journal"]
    if include_abstract:
        source_fields.append("abstract")

    should_clauses = [
        {
            "multi_match": {
                "query":  term,
                "fields": ["title.exact^2", "abstract.exact"],
                "type":   "phrase",
            }
        }
        for term in clean_terms
    ]

    body = {
        "query": {
            "bool": {
                "should":               should_clauses,
                "minimum_should_match": 1,
            }
        },
        "_source": source_fields,
        "size":    batch_size,
        "sort":    [{"_score": "desc"}, {"pmid": "asc"}],
    }

    best: dict[int, dict] = {}

    while True:
        res  = es.search(index=index, body=body)
        hits = res["hits"]["hits"]
        if not hits:
            break

        for hit in hits:
            src   = hit["_source"]
            pmid  = src.get("pmid")
            score = hit["_score"]
            if pmid is None:
                continue
            if pmid not in best or score > best[pmid]["es_score"]:
                entry: dict = {
                    "pmid":     int(pmid),
                    "title":    src.get("title", ""),
                    "year":     src.get("year"),
                    "journal":  src.get("journal", ""),
                    "es_score": score,
                }
                if include_abstract:
                    entry["abstract"] = src.get("abstract", "")
                best[pmid] = entry

        if len(hits) < batch_size:
            break
        body["search_after"] = hits[-1]["sort"]

    return list(best.values())


def search_node_tracked(
    es,
    index:            str,
    primary_term:     str,
    alias_terms:      list[str],
    batch_size:       int = 1000,
    include_abstract: bool = False,
) -> dict[int, dict]:
    """
    Search with a primary term and alias terms, recording which matched each PMID.

    PMIDs found by primary_term get matched_by="primary".
    PMIDs found only by alias_terms get matched_by="alias".
    If a PMID is found by both, "primary" takes precedence.

    Args:
        es:               Elasticsearch client
        index:            index name
        primary_term:     canonical search term (e.g. official gene symbol)
        alias_terms:      list of alias terms (synonyms, alternative symbols)
        batch_size:       hits per page
        include_abstract: if True, include abstract text in results

    Returns:
        {pmid: {pmid, title, year, journal, es_score, matched_by[, abstract]}}
    """
    primary_hits = search_node_with_terms(
        es, index, [primary_term], batch_size, include_abstract
    )
    result: dict[int, dict] = {
        h["pmid"]: {**h, "matched_by": "primary"} for h in primary_hits
    }

    if alias_terms:
        alias_hits = search_node_with_terms(
            es, index, alias_terms, batch_size, include_abstract
        )
        for h in alias_hits:
            pmid = h["pmid"]
            if pmid not in result:
                result[pmid] = {**h, "matched_by": "alias"}

    return result
