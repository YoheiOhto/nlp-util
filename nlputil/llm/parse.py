from __future__ import annotations
import json
import re


def parse_json_output(raw_text: str) -> list[dict]:
    """Extract a JSON array from raw LLM output.

    Handles markdown code fences and extracts the first [...] block.

    Returns:
        Parsed list of dicts, or [] on failure.
    """
    text = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return []


def parse_json_object(raw_text: str) -> dict:
    """Extract a JSON object from raw LLM output.

    Returns:
        Parsed dict, or {} on failure.
    """
    text = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}
