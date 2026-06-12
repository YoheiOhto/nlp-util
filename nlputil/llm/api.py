from __future__ import annotations
import time


def chat_completion(
    prompt: str,
    model: str,
    client,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    system: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    """Single-prompt completion using an OpenAI-compatible client.

    Works with openai.OpenAI, vLLM's OpenAI server, together.ai, etc.

    Args:
        prompt: User message.
        model: Model name or path.
        client: openai.OpenAI (or compatible) client instance.
        max_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        system: Optional system message.
        max_retries: Retry attempts on transient errors.
        retry_delay: Base seconds between retries (linearly scaled per attempt).

    Returns:
        Generated text string.
    """
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay * (attempt + 1))
    raise RuntimeError("unreachable")


def batch_chat_completion(
    prompts: list[str],
    model: str,
    client,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    system: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> list[str]:
    """Run chat_completion for each prompt sequentially.

    For high-throughput inference use nlputil.llm.vllm.generate_batch instead.
    """
    return [
        chat_completion(p, model, client, max_tokens, temperature, system, max_retries, retry_delay)
        for p in prompts
    ]


def claude_completion(
    prompt: str,
    model: str,
    client,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    system: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    """Single-prompt completion using an Anthropic client.

    Args:
        prompt: User message.
        model: Anthropic model name (e.g. 'claude-sonnet-4-6').
        client: anthropic.Anthropic client instance.
        max_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        system: Optional system message.
        max_retries: Retry attempts on transient errors.
        retry_delay: Base seconds between retries.

    Returns:
        Generated text string.
    """
    for attempt in range(max_retries):
        try:
            kwargs: dict = dict(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if system:
                kwargs["system"] = system
            response = client.messages.create(**kwargs)
            return response.content[0].text.strip()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay * (attempt + 1))
    raise RuntimeError("unreachable")
