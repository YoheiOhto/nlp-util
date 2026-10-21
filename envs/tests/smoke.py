"""Smoke test for an nlp-envs environment, run inside its SIF.

Usage (inside the container; envs/test.sh does this):
    python /opt/nlp-envs/smoke.py --env llm [--no-gpu] [--vllm-model Qwen/Qwen3-0.6B]

It checks that every installed package has exactly the version in uv.lock, that
the environment holds what its name promises (and nothing from the others), and,
unless --no-gpu, that the GPU really computes: a matmul, a tiny transformer, and
for llm/train the vLLM import (optionally a short generation), a LoRA step and an
8-bit layer. Every check prints PASS or FAIL; the exit status is 1 if any fails.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import platform
import re
import sys
import tomllib
import traceback
from collections.abc import Callable
from pathlib import Path

# Packages each environment must import, and packages it must not contain.
REQUIRED = {
    "nlp": ["torch", "transformers", "tokenizers", "datasets", "accelerate", "numpy", "pandas",
            "polars", "duckdb", "sklearn", "spacy", "elasticsearch"],
    "llm": ["vllm", "openai", "anthropic", "tiktoken"],
    "train": ["peft", "trl", "bitsandbytes", "wandb"],
}
EXCLUSIVE = {"llm": ["vllm"], "train": ["peft", "trl", "bitsandbytes"]}


def canonical(name: str) -> str:
    """Normalize a distribution name the way uv.lock writes it (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def find_lock() -> Path:
    """Return uv.lock next to this script (in the SIF) or one folder up (in the repository)."""
    here = Path(__file__).resolve().parent
    for candidate in (here / "uv.lock", here.parent / "uv.lock"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("uv.lock not found next to smoke.py or in its parent folder")


def locked_versions(lock: Path) -> dict[str, str]:
    """Map each package in uv.lock to its locked version."""
    data = tomllib.loads(lock.read_text(encoding="utf-8"))
    return {canonical(p["name"]): p["version"] for p in data.get("package", []) if "version" in p}


class Checker:
    """Collects PASS/FAIL results so one failure does not hide the others."""

    def __init__(self) -> None:
        self.failed = 0

    def run(self, name: str, check: Callable[[], str]) -> None:
        """Run one check and print its result.

        Args:
            name: A short label.
            check: Returns a detail string, or raises on failure.
        """
        try:
            detail = check()
            print(f"PASS {name}: {detail}", flush=True)
        except Exception as exc:  # noqa: BLE001 - every failure must be reported, whatever it is
            self.failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=3)


def check_python() -> str:
    assert sys.version_info[:2] == (3, 12), f"Python {platform.python_version()}, expected 3.12"
    return f"Python {platform.python_version()} on {platform.machine()}"


def check_lock(lock: Path) -> str:
    """Every installed distribution that uv.lock knows must have the locked version."""
    locked = locked_versions(lock)
    wrong = []
    checked = 0
    for dist in metadata.distributions():
        name = canonical(dist.metadata["Name"])
        if name in locked:
            checked += 1
            if dist.version != locked[name]:
                wrong.append(f"{name} {dist.version} (lock {locked[name]})")
    assert not wrong, "versions differ from uv.lock: " + ", ".join(sorted(wrong))
    assert checked > 50, f"only {checked} packages found; is this the right environment?"
    return f"{checked} installed packages match {lock}"


def check_contents(env: str) -> str:
    """The environment holds its own packages and none of the other environments' extras."""
    wanted = REQUIRED["nlp"] + (REQUIRED.get(env, []) if env != "nlp" else [])
    for module in wanted:
        importlib.import_module(module)
    for other, modules in EXCLUSIVE.items():
        if other == env:
            continue
        for module in modules:
            try:
                metadata.version(module)
            except metadata.PackageNotFoundError:
                continue
            raise AssertionError(f"{module} belongs to '{other}' but is installed in '{env}'")
    return f"imported {len(wanted)} modules"


def check_gpu() -> str:
    import torch

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False (driver? --nv?)"
    assert (torch.version.cuda or "").startswith("13"), f"torch built for CUDA {torch.version.cuda}, expected 13.x"
    major, minor = torch.cuda.get_device_capability(0)
    arches = torch.cuda.get_arch_list()
    # A cubin for sm_XY also runs on sm_XZ (Z >= Y) of the same major, e.g. sm_120 on GB10 (sm_121).
    usable = [a for a in arches if a.startswith(("sm_", "compute_")) and int(a.split("_")[1]) // 10 == major
              and int(a.split("_")[1]) % 10 <= minor]
    assert usable, f"no kernels for sm_{major}{minor} in {arches}"
    a = torch.randn(512, 512, device="cuda", dtype=torch.float32)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float32)
    on_gpu = (a @ b).cpu()
    on_cpu = a.cpu() @ b.cpu()
    assert torch.allclose(on_gpu, on_cpu, rtol=1e-3, atol=1e-2), "GPU matmul differs from CPU"
    return (f"{torch.cuda.get_device_name(0)} sm_{major}{minor}, torch {torch.__version__} "
            f"CUDA {torch.version.cuda}, kernels {usable}")


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def check_transformers() -> str:
    import torch
    from transformers import BertConfig, BertModel

    config = BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64)
    model = BertModel(config).to(device()).eval()
    with torch.no_grad():
        out = model(input_ids=torch.randint(0, 100, (2, 16), device=device()))
    assert tuple(out.last_hidden_state.shape) == (2, 16, 32)
    return f"tiny BERT forward on {device()}"


def check_text_tools() -> str:
    import duckdb
    import polars as pl
    import spacy

    doc = spacy.blank("en")("The patient received 5 mg of aspirin.")
    assert len(doc) == 8, f"unexpected tokens {[t.text for t in doc]}"
    frame = pl.DataFrame({"x": [1, 2, 3]})
    total = duckdb.sql("SELECT sum(x) AS s FROM frame").fetchone()[0]
    assert total == 6
    return "spaCy tokenizer, polars + duckdb"


def check_vllm(model: str | None) -> str:
    import vllm

    if not model:
        return f"vllm {vllm.__version__} imported (no model given)"
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, gpu_memory_utilization=0.5, max_model_len=512, enforce_eager=True)
    text = llm.generate(["Aspirin is"], SamplingParams(max_tokens=8, temperature=0.0))[0].outputs[0].text
    assert text.strip(), "empty generation"
    return f"vllm {vllm.__version__} generated {text!r} with {model}"


def check_train() -> str:
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(GPT2Config(vocab_size=100, n_positions=32, n_embd=32, n_layer=2, n_head=2))
    model = get_peft_model(model, LoraConfig(r=4, target_modules=["c_attn"], fan_in_fan_out=True)).to(device())
    ids = torch.randint(0, 100, (2, 16), device=device())
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3).step()
    detail = f"LoRA step on {device()} (loss {loss.item():.3f})"
    if device() == "cuda":
        layer = bnb.nn.Linear8bitLt(32, 32, has_fp16_weights=False).cuda()
        out = layer(torch.randn(4, 32, device="cuda", dtype=torch.float16))
        assert torch.isfinite(out).all(), "8-bit layer produced non-finite values"
        detail += ", bitsandbytes 8-bit layer"
    return detail


def main(argv: list[str] | None = None) -> int:
    """Run the checks for one environment.

    Args:
        argv: Command-line arguments (without the program name).

    Returns:
        0 when every check passes, otherwise 1.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", required=True, choices=sorted(REQUIRED))
    parser.add_argument("--no-gpu", action="store_true", help="skip GPU checks (used while building)")
    parser.add_argument("--vllm-model", help="also generate with this model (llm only; downloads it)")
    args = parser.parse_args(argv)

    checker = Checker()
    lock = find_lock()
    checker.run("python", check_python)
    checker.run("uv.lock", lambda: check_lock(lock))
    checker.run(f"contents of {args.env}", lambda: check_contents(args.env))
    checker.run("text tools", check_text_tools)
    if not args.no_gpu:
        checker.run("gpu", check_gpu)
    checker.run("transformers", check_transformers)
    if args.env == "llm":
        checker.run("vllm", lambda: check_vllm(None if args.no_gpu else args.vllm_model))
    if args.env == "train":
        checker.run("train", check_train)
    print(f"{'FAILED' if checker.failed else 'OK'}: {args.env} ({checker.failed} failed)", flush=True)
    return 1 if checker.failed else 0


if __name__ == "__main__":
    sys.exit(main())
