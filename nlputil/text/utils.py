from __future__ import annotations
import re


def clean_text(text: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    """Truncate text to max_chars, appending suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def bio_to_spans(
    tokens: list[str],
    tags: list[str],
) -> list[tuple[str, str, int, int]]:
    """Convert BIO tag sequence to span list.

    Returns:
        List of (entity_text, label, start_idx, end_idx) tuples.
    """
    spans: list[tuple[str, str, int, int]] = []
    start: int | None = None
    current_label: str | None = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None:
                spans.append((" ".join(tokens[start:i]), current_label, start, i))
            start = i
            current_label = tag[2:]
        elif tag.startswith("I-") and start is not None and tag[2:] == current_label:
            pass
        else:
            if start is not None:
                spans.append((" ".join(tokens[start:i]), current_label, start, i))
                start = None
                current_label = None

    if start is not None:
        spans.append((" ".join(tokens[start:]), current_label, start, len(tokens)))

    return spans


def spans_to_bio(
    tokens: list[str],
    spans: list[tuple[str, str, int, int]],
) -> list[str]:
    """Convert span list back to BIO tag sequence.

    Args:
        tokens: List of tokens.
        spans:  List of (entity_text, label, start_idx, end_idx) tuples.

    Returns:
        BIO tag list of the same length as tokens.
    """
    tags = ["O"] * len(tokens)
    for _, label, start, end in spans:
        for i in range(start, end):
            tags[i] = f"B-{label}" if i == start else f"I-{label}"
    return tags
