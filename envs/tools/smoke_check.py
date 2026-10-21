"""Smoke check for one nlp-util environment profile.

Imports the profile's key packages, records their versions, and (when a GPU is
visible) runs a small CUDA computation so that a "built" environment is also a
"working" environment. Import success alone does not prove GPU compatibility,
so the GPU part runs real kernels and checks the result numerically.

The report is written as JSON; the process exits non-zero on any failure.

Example:
    python smoke_check.py --profile serve-vllm --report report.json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

# Modules that must import in each profile (import name -> distribution name).
PROFILE_IMPORTS: dict[str, dict[str, str]] = {
    "base": {
        "torch": "torch",
        "transformers": "transformers",
        "datasets": "datasets",
        "sentence_transformers": "sentence-transformers",
        "polars": "polars",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "spacy": "spacy",
        "elasticsearch": "elasticsearch",
    },
    "bio-scispacy": {
        "numpy": "numpy",
        "spacy": "spacy",
        "scispacy": "scispacy",
        "nmslib": "nmslib-metabrainz",
    },
    "serve-vllm": {
        "torch": "torch",
        "transformers": "transformers",
        "vllm": "vllm",
        "flashinfer": "flashinfer-python",
        "lm_eval": "lm-eval",
    },
    "serve-vllm-0.26": {
        "torch": "torch",
        "transformers": "transformers",
        "vllm": "vllm",
        "flashinfer": "flashinfer-python",
    },
    "serve-sglang": {
        "torch": "torch",
        "transformers": "transformers",
        "sglang": "sglang",
        "sgl_kernel": "sglang-kernel",
        "flashinfer": "flashinfer-python",
    },
    "train-hf": {
        "torch": "torch",
        "transformers": "transformers",
        "trl": "trl",
        "peft": "peft",
        "accelerate": "accelerate",
        "deepspeed": "deepspeed",
        "liger_kernel": "liger-kernel",
        "kernels": "kernels",
        "bitsandbytes": "bitsandbytes",
    },
    "train-unsloth": {
        # unsloth must be imported before transformers/trl/peft to patch them.
        "unsloth": "unsloth",
        "torch": "torch",
        "transformers": "transformers",
        "trl": "trl",
        "peft": "peft",
        "bitsandbytes": "bitsandbytes",
    },
    "allinone": {
        # torch first (needed to decide whether unsloth can be imported), then
        # unsloth before transformers/trl/peft so that it can patch them.
        "torch": "torch",
        "unsloth": "unsloth",
        "transformers": "transformers",
        "vllm": "vllm",
        "flashinfer": "flashinfer-python",
        "trl": "trl",
        "peft": "peft",
        "accelerate": "accelerate",
        "deepspeed": "deepspeed",
        "liger_kernel": "liger-kernel",
        "kernels": "kernels",
        "bitsandbytes": "bitsandbytes",
        "lm_eval": "lm-eval",
        "datasets": "datasets",
        "sentence_transformers": "sentence-transformers",
        "polars": "polars",
        "sklearn": "scikit-learn",
        "spacy": "spacy",
    },
}

# Optional modules: recorded when present, never a failure when absent.
OPTIONAL_IMPORTS: dict[str, dict[str, str]] = {
    "train-hf": {"flash_attn": "flash-attn"},
    "train-unsloth": {"xformers": "xformers"},
    "allinone": {"flash_attn": "flash-attn", "xformers": "xformers"},
}

# Modules that refuse to import without a CUDA device (unsloth raises
# "cannot find any torch accelerator"). In a CPU-only build job they are
# recorded as skipped; with --require-gpu a missing GPU already fails.
GPU_ONLY_IMPORTS: frozenset[str] = frozenset({"unsloth"})

# Profiles that get the tiny LoRA training-step check on the GPU.
TRAINING_PROFILES: frozenset[str] = frozenset({"train-hf", "allinone"})


def _version(dist: str) -> str | None:
    """Return the installed version of a distribution, or None if absent."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run(name: str, fn: Callable[[], Any], results: dict[str, Any]) -> bool:
    """Run one check, storing its value or the formatted exception."""
    start = time.perf_counter()
    try:
        value = fn()
        results[name] = {"ok": True, "value": value}
        ok = True
    except Exception:  # noqa: BLE001 - the report must capture any failure
        results[name] = {"ok": False, "error": traceback.format_exc(limit=5)}
        ok = False
    results[name]["seconds"] = round(time.perf_counter() - start, 3)
    return ok


def _gpu_checks(require_gpu: bool, results: dict[str, Any]) -> bool:
    """Run CUDA checks; returns False on failure or when a required GPU is absent."""
    import torch

    if not torch.cuda.is_available():
        results["cuda_available"] = {"ok": not require_gpu, "value": False}
        return not require_gpu

    def device_info() -> dict[str, Any]:
        props = torch.cuda.get_device_properties(0)
        return {
            "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "total_memory_gib": round(props.total_memory / 2**30, 1),
            "arch_list": torch.cuda.get_arch_list(),
        }

    def matmul_bf16() -> float:
        # Compare a bf16 GPU matmul with a float64 CPU reference.
        gen = torch.Generator().manual_seed(0)
        a = torch.randn(512, 512, generator=gen, dtype=torch.float64)
        b = torch.randn(512, 512, generator=gen, dtype=torch.float64)
        ref = a @ b
        got = (a.to("cuda", torch.bfloat16) @ b.to("cuda", torch.bfloat16)).double().cpu()
        rel = float((got - ref).norm() / ref.norm())
        if rel > 2e-2:
            raise AssertionError(f"bf16 matmul relative error too large: {rel}")
        return rel

    def sdpa() -> list[int]:
        q = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
        out = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        torch.cuda.synchronize()
        if not torch.isfinite(out).all():
            raise AssertionError("SDPA produced non-finite values")
        return list(out.shape)

    ok = _run("cuda_device", device_info, results)
    ok &= _run("cuda_matmul_bf16", matmul_bf16, results)
    ok &= _run("cuda_sdpa", sdpa, results)
    return ok


def _cuda_available() -> bool:
    """Return whether torch sees a CUDA device.

    A failing torch import returns False here; the "import:torch" check
    reports the actual error.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - reported by the import check instead
        return False
    return bool(torch.cuda.is_available())


def _flash_attn_check() -> dict[str, Any]:
    """Compare flash_attn_func with an fp32 SDPA reference (causal, bf16 inputs).

    Returns:
        Version and max absolute errors of flash-attn and of bf16 SDPA, both
        against the same fp32 reference.

    Raises:
        AssertionError: if the flash-attn error exceeds the bf16 tolerance or
            its backward pass produces non-finite gradients.
    """
    import torch
    import torch.nn.functional as F
    from flash_attn import flash_attn_func

    gen = torch.Generator(device="cuda").manual_seed(0)
    batch, seq, heads, dim = 2, 256, 8, 64
    # flash-attn layout is (batch, seq, heads, dim); SDPA is (batch, heads, seq, dim).
    q, k, v = (
        torch.randn(batch, seq, heads, dim, generator=gen, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    q.requires_grad_(True)
    out = flash_attn_func(q, k, v, causal=True)

    def sdpa(dtype: torch.dtype) -> torch.Tensor:
        res = F.scaled_dot_product_attention(
            *(t.detach().to(dtype).transpose(1, 2) for t in (q, k, v)), is_causal=True
        )
        return res.transpose(1, 2).float()

    ref = sdpa(torch.float32)
    flash_err = float((out.float() - ref).abs().max())
    sdpa_err = float((sdpa(torch.bfloat16) - ref).abs().max())
    # bf16 has ~3 significant digits; outputs here are O(1).
    tol = max(2e-2, 2.0 * sdpa_err)
    if not flash_err <= tol:
        raise AssertionError(f"flash-attn max abs error {flash_err} > {tol} (bf16 SDPA: {sdpa_err})")
    out.float().sum().backward()
    torch.cuda.synchronize()
    if q.grad is None or not torch.isfinite(q.grad).all():
        raise AssertionError("flash-attn backward produced missing or non-finite gradients")
    return {
        "version": _version("flash-attn"),
        "shape_bshd": [batch, seq, heads, dim],
        "max_abs_err_flash_vs_fp32": flash_err,
        "max_abs_err_sdpa_bf16_vs_fp32": sdpa_err,
        "tolerance": tol,
    }


def _train_step_check(attn_implementation: str) -> dict[str, Any]:
    """Run one LoRA training step on a tiny random Llama model (no download).

    Args:
        attn_implementation: transformers attention backend ("sdpa" or
            "flash_attention_2").

    Returns:
        Loss before the step, trainable parameter count and the LoRA update norm.

    Raises:
        AssertionError: on a non-finite loss or when the step does not change
            the LoRA weights.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attn_implementation=attn_implementation,
    )
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            model = LlamaForCausalLM(config)
    finally:
        torch.set_default_dtype(previous)
    model = get_peft_model(
        model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise AssertionError("LoRA produced no trainable parameters")
    before = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    input_ids = torch.randint(0, config.vocab_size, (2, 64), device="cuda")
    model.train()
    loss = model(input_ids=input_ids, labels=input_ids).loss
    if not torch.isfinite(loss):
        raise AssertionError(f"non-finite training loss: {loss.item()}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    delta = float(sum((p.detach().float() - b.float()).norm() for p, b in zip(trainable, before)))
    if not delta > 0.0:
        raise AssertionError("optimizer step did not change the LoRA weights")
    return {
        "attn_implementation": attn_implementation,
        "loss": float(loss.item()),
        "trainable_params": int(sum(p.numel() for p in trainable)),
        "lora_update_norm": delta,
    }


def main() -> int:
    """Parse arguments, run the checks and write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", required=True, choices=sorted(PROFILE_IMPORTS))
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="fail when no CUDA device is visible (use inside GPU allocations)",
    )
    args = parser.parse_args()

    results: dict[str, Any] = {}
    ok = True
    imports = PROFILE_IMPORTS[args.profile]
    for module, dist in imports.items():
        if module in GPU_ONLY_IMPORTS and "torch" in imports and not args.require_gpu:
            # torch precedes GPU-only modules in PROFILE_IMPORTS.
            if not _cuda_available():
                results[f"import:{module}"] = {
                    "ok": True,
                    "value": {"skipped": "needs a CUDA device", "version": _version(dist)},
                    "seconds": 0.0,
                }
                continue
        ok &= _run(
            f"import:{module}",
            lambda m=module, d=dist: {"module": importlib.import_module(m).__name__, "version": _version(d)},
            results,
        )
    optional = {dist: _version(dist) for dist in OPTIONAL_IMPORTS.get(args.profile, {}).values()}
    flash_installed = optional.get("flash-attn") is not None
    if flash_installed:
        # An ABI mismatch with torch shows up at import time, GPU or not.
        ok &= _run(
            "import:flash_attn",
            lambda: {"module": importlib.import_module("flash_attn").__name__, "version": _version("flash-attn")},
            results,
        )
    # CPU-only profiles (no torch) have nothing to run on the GPU.
    if ok and "torch" in imports:
        gpu_ok = _gpu_checks(args.require_gpu, results)
        ok &= gpu_ok
        if gpu_ok and _cuda_available():
            if flash_installed:
                ok &= _run("cuda_flash_attn_vs_sdpa", _flash_attn_check, results)
            if args.profile in TRAINING_PROFILES:
                ok &= _run("cuda_lora_train_step:sdpa", lambda: _train_step_check("sdpa"), results)
                if flash_installed:
                    ok &= _run(
                        "cuda_lora_train_step:flash_attention_2",
                        lambda: _train_step_check("flash_attention_2"),
                        results,
                    )

    report = {
        "profile": args.profile,
        "ok": bool(ok),
        "host": socket.gethostname(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "optional_versions": optional,
        "checks": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for name, res in results.items():
        status = "ok " if res["ok"] else "NG "
        detail = res.get("value") if res["ok"] else res["error"].strip().splitlines()[-1]
        print(f"[{status}] {name}: {detail}")
    print(f"profile={args.profile} ok={ok} report={args.report}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
