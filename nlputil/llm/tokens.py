from __future__ import annotations


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens using a HuggingFace tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def fits_in_context(text: str, tokenizer, max_tokens: int) -> bool:
    """Return True if text fits within max_tokens."""
    return count_tokens(text, tokenizer) <= max_tokens


def truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Truncate text so it fits within max_tokens.

    Uses the tokenizer's decode, so round-trip fidelity is tokenizer-dependent.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def batch_token_counts(texts: list[str], tokenizer) -> list[int]:
    """Count tokens for a batch of texts (faster than calling count_tokens in a loop)."""
    encoded = tokenizer(texts, add_special_tokens=False)
    return [len(ids) for ids in encoded["input_ids"]]
