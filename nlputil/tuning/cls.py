from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from transformers import PreTrainedTokenizerBase


def tokenize_for_cls(
    examples: dict,
    tokenizer: "PreTrainedTokenizerBase",
    max_length: int = 512,
    text_col: str = "text",
    label_col: str = "label",
) -> dict:
    """Tokenize text for sequence classification.

    Designed for datasets.Dataset.map().

    Args:
        examples: Batch from a HuggingFace dataset.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum token sequence length.
        text_col: Column name containing raw text.
        label_col: Column name containing integer labels.

    Returns:
        Dict with input_ids, attention_mask, and labels.
    """
    tokenized = tokenizer(
        examples[text_col],
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    tokenized["labels"] = examples[label_col]
    return tokenized


def compute_cls_metrics(eval_pred) -> dict[str, float]:
    """Compute accuracy and macro-F1 for sequence classification.

    Designed for transformers.Trainer(compute_metrics=...).
    """
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1":       float(f1_score(labels, preds, average="macro")),
    }


def format_cls_predictions(
    trainer,
    dataset,
    id2label: dict[int, str],
) -> "pd.DataFrame":
    """Run inference and return a per-sample prediction DataFrame.

    Returns:
        DataFrame with columns [true_label, pred_label, confidence].
    """
    import pandas as pd

    output = trainer.predict(dataset)
    preds = np.argmax(output.predictions, axis=-1)
    logits = output.predictions.astype(float)
    # stable softmax
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)

    return pd.DataFrame({
        "true_label": [id2label[int(l)] for l in output.label_ids],
        "pred_label": [id2label[int(p)] for p in preds],
        "confidence": probs.max(axis=-1),
    })
