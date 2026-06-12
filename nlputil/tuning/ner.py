from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from transformers import PreTrainedTokenizerBase


def tokenize_and_align_labels(
    examples: dict,
    tokenizer: "PreTrainedTokenizerBase",
    label2id: dict[str, int],
    max_length: int = 512,
    label_col: str = "ner_tags",
    token_col: str = "tokens",
) -> dict:
    """Tokenize word-level NER data and align BIO labels to subword tokens.

    Designed for datasets.Dataset.map(). Labels for non-first subword tokens
    and special tokens are set to -100 so the loss ignores them.

    Args:
        examples: Batch from a HuggingFace dataset.
        tokenizer: HuggingFace tokenizer.
        label2id: Mapping from BIO label string to integer id.
        max_length: Maximum subword token sequence length.
        label_col: Column name containing BIO label strings or ints.
        token_col: Column name containing word tokens.

    Returns:
        Dict with input_ids, attention_mask, and labels.
    """
    tokenized = tokenizer(
        examples[token_col],
        is_split_into_words=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    id2label_str = {v: k for k, v in label2id.items()}
    all_labels: list[list[int]] = []

    for i, raw_labels in enumerate(examples[label_col]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned: list[int] = []
        prev_word_id: int | None = None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)
            elif wid != prev_word_id:
                raw = raw_labels[wid]
                label_str = raw if isinstance(raw, str) else id2label_str.get(raw, "O")
                aligned.append(label2id.get(label_str, -100))
            else:
                aligned.append(-100)
            prev_word_id = wid
        all_labels.append(aligned)

    tokenized["labels"] = all_labels
    return tokenized


def compute_ner_metrics(eval_pred, label_list: list[str]) -> dict[str, float]:
    """Compute seqeval span-level metrics for NER.

    Designed for transformers.Trainer(compute_metrics=...).

    Args:
        eval_pred: (logits, labels) tuple from Trainer.
        label_list: Ordered list of BIO label strings (index = label id).

    Returns:
        Dict with precision, recall, f1.
    """
    from seqeval.metrics import f1_score, precision_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_seqs: list[list[str]] = []
    pred_seqs: list[list[str]] = []
    for pred_row, label_row in zip(preds, labels):
        true_seq, pred_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l != -100:
                true_seq.append(label_list[l])
                pred_seq.append(label_list[p])
        true_seqs.append(true_seq)
        pred_seqs.append(pred_seq)

    return {
        "precision": precision_score(true_seqs, pred_seqs),
        "recall":    recall_score(true_seqs, pred_seqs),
        "f1":        f1_score(true_seqs, pred_seqs),
    }


def format_ner_predictions(
    trainer,
    dataset,
    tokenizer: "PreTrainedTokenizerBase",
    id2label: dict[int, str],
) -> "pd.DataFrame":
    """Run inference and return a per-token prediction DataFrame.

    Returns:
        DataFrame with columns [token, true_label, pred_label].
    """
    import pandas as pd

    output = trainer.predict(dataset)
    preds = np.argmax(output.predictions, axis=-1)
    rows: list[dict] = []
    for pred_row, label_row, input_row in zip(preds, output.label_ids, dataset["input_ids"]):
        for p, l, t in zip(pred_row, label_row, input_row):
            if l == -100:
                continue
            rows.append({
                "token":      tokenizer.decode([t]),
                "true_label": id2label[l],
                "pred_label": id2label[p],
            })
    return pd.DataFrame(rows)
