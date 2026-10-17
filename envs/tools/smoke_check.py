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
        "scispacy": "scispacy",
        "elasticsearch": "elasticsearch",
    },
    "serve-vllm": {
        "torch": "torch",
        "transformers": "transformers",
        "vllm": "vllm",
        "flashinfer": "flashinfer-python",
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
}

# Optional modules: recorded when present, never a failure when absent.
OPTIONAL_IMPORTS: dict[str, dict[str, str]] = {
    "train-hf": {"flash_attn": "flash-attn"},
    "train-unsloth": {"xformers": "xformers"},
}


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
    for module, dist in PROFILE_IMPORTS[args.profile].items():
        ok &= _run(
            f"import:{module}",
            lambda m=module, d=dist: {"module": importlib.import_module(m).__name__, "version": _version(d)},
            results,
        )
    optional = {dist: _version(dist) for dist in OPTIONAL_IMPORTS.get(args.profile, {}).values()}
    if ok:
        ok &= _gpu_checks(args.require_gpu, results)

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
