"""
vLLM inference utilities.

Provides a robust JSON extractor for LLM outputs and a batch inference
helper that applies chat templates before calling llm.generate().

Note: vllm and transformers are only imported at call time (TYPE_CHECKING guard),
so this module can be imported on CPU-only login nodes without errors.
"""

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


def parse_json_output(raw_text: str) -> list[dict]:
    """
    Extract a JSON array from LLM output text.

    Handles three common formats:
      1. Bare JSON array:          ``[{"key": "val"}, ...]``
      2. Markdown code block:      ```json\\n[...]\\n```
      3. Array embedded in prose:  ``Here are the results: [...] Done.``

    Returns an empty list if parsing fails, so callers can always iterate
    the result without a try/except.

    Args:
        raw_text: raw string output from an LLM

    Returns:
        list of dicts parsed from the JSON array, or [] on failure
    """
    json_text = raw_text

    md_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
    if md_match:
        json_text = md_match.group(1).strip()
    else:
        bracket_match = re.search(r"\[[\s\S]*\]", raw_text)
        if bracket_match:
            json_text = bracket_match.group(0)

    try:
        parsed = json.loads(json_text)
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        return []


def generate_batch_vllm(
    prompts:         list[str],
    llm:             "LLM",
    processor,
    sampling_params: "SamplingParams",
) -> list[tuple[list[dict], str]]:
    """
    Run batch inference with vLLM, applying the model's chat template first.

    Applies processor.apply_chat_template to each prompt before calling
    llm.generate(). Each output is passed through parse_json_output so
    the caller receives structured results alongside the raw text.

    Args:
        prompts:         list of user-turn prompt strings
        llm:             vLLM LLM instance
        processor:       tokenizer/processor with apply_chat_template
        sampling_params: vLLM SamplingParams instance

    Returns:
        list of (parsed_json, raw_text) tuples, one per input prompt
    """
    formatted = [
        processor.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    outputs = llm.generate(formatted, sampling_params)
    return [
        (parse_json_output(o.outputs[0].text.strip()), o.outputs[0].text.strip())
        for o in outputs
    ]
