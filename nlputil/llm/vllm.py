"""
vLLM batch inference utility.

Applies chat templates and runs llm.generate() in batch.

Note: vllm is only imported at call time (TYPE_CHECKING guard),
so this module can be imported on CPU-only login nodes without errors.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


def generate_batch(
    prompts:         list[str],
    llm:             "LLM",
    processor,
    sampling_params: "SamplingParams",
) -> list[str]:
    """
    Run batch inference with vLLM, applying the model's chat template first.

    Args:
        prompts:         list of user-turn prompt strings
        llm:             vLLM LLM instance
        processor:       tokenizer/processor with apply_chat_template
        sampling_params: vLLM SamplingParams instance

    Returns:
        list of raw output strings, one per input prompt
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
    return [o.outputs[0].text.strip() for o in outputs]
