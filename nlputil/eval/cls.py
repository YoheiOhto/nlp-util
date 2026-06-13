from __future__ import annotations


def classification_metrics(
    y_true: list,
    y_pred: list,
    labels: list | None = None,
) -> dict[str, object]:
    """Compute precision, recall, F1, and accuracy using sklearn.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels.
        labels: Optional label order; defaults to sorted unique values.

    Returns:
        Flat dict with macro precision/recall/f1, accuracy, and full report dict.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "precision": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "report":    classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }
