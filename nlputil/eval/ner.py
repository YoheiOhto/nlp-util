from __future__ import annotations


def span_f1(
    true_sequences: list[list[str]],
    pred_sequences: list[list[str]],
) -> dict[str, object]:
    """Compute span-level NER metrics using seqeval.

    Args:
        true_sequences: List of BIO tag sequences (gold).
        pred_sequences: List of BIO tag sequences (predicted).

    Returns:
        Dict with precision, recall, f1, and per-type report dict.
    """
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "precision": precision_score(true_sequences, pred_sequences),
        "recall":    recall_score(true_sequences, pred_sequences),
        "f1":        f1_score(true_sequences, pred_sequences),
        "report":    classification_report(true_sequences, pred_sequences, output_dict=True),
    }
