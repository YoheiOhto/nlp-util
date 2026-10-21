"""0005 analyze_engine_comparison: speed tables (0003) and hidden-state fidelity (0004).

Everything is recomputed from the raw files of 0003 and 0004 (plan.evaluation.quality_checks.recompute):

Speed (0003): ``outputs/0003_20261019_eval_engine_speed/<model>__<format>__r<k>/``
  - ``<workload_id>.json``: raw ``vllm bench serve --save-result`` file (vLLM 0.31).
    A repeat is valid when ``failed == 0``, ``completed == num_prompts`` of the workload and
    every request returned ``output_len`` (256) tokens (0003 validity rule).
    Per-request lists (``ttfts``, ``itls``, ``latencies``, ``output_lens``, ``input_lens``;
    present with ``--save-detailed``) are used to recompute TTFT/TPOT; without them the
    file's own summary fields are used and the row is marked ``metric_source=saved_summary``.
  - ``server_meta.json``: ``load_time_s``, ``peak_vram_mib``.
  - ``llama_bench.json`` (llama.cpp tasks): ``llama-bench -o json`` records.
  Cells are aggregated as median and min-max over valid repeats; fewer than
  ``min_valid_repeats`` (2) valid repeats -> ``status = incomplete`` (no value shown as 0).

Fidelity (0004): ``outputs/0004_20261019_extract_hidden_states/<variant>/<prompt_id>/<layer>.npy``
  - per token: cosine similarity and relative L2 ``||x - x_ref|| / ||x_ref||`` (float64) vs
    ``hf-bf16``; layers pair by identical name (pre-norm with pre-norm, normed with normed;
    ``final_norm_cb`` -> ``final_norm``); statistics over tokens pooled across prompts,
    for all tokens and excluding position 0.
  - last-token logits: top-1 agreement, top-10 overlap, KL(p_ref || p).
  - sanity: ``hf-bf16`` vs itself gives cosine 1 and error 0 (asserted).
  - independent check: 3 random (variant, prompt, layer) triples (``default_rng(0)``) are
    recomputed with an independent pure-Python ``math.fsum`` path; agreement to 1e-6 is asserted.

Outputs in ``outputs/0005_20261019_analyze_engine_comparison/all/``: ``speed_summary.parquet``,
``speed_repeats.parquet``, ``fidelity_by_layer.parquet``, ``logits_agreement.parquet``,
``capability.parquet``, ``checks.json`` and ``figures/*.png`` each with ``*.source.parquet``
and ``figures/manifest.json`` (script sha256 and git revision).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

VARIANT_KEY = "all"


# ── Generic helpers ───────────────────────────────────────────────────────────


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    """Set up a logger writing to both console and run_dir/experiment.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_config(exp_dir: Path) -> dict[str, Any]:
    """Load config.yml from the experiment directory (must exist)."""
    path = exp_dir / "config.yml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} is not a mapping")
    return cfg


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def git_revision(project_root: Path) -> str:
    """Return the repository HEAD commit, or 'unknown'."""
    try:
        out = subprocess.run(["git", "-C", str(project_root), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def read_completion(variant_dir: Path) -> dict[str, Any]:
    """Return completion info of an upstream variant (status, guard_skipped_at, job_id)."""
    path = variant_dir / "completion.json"
    if not path.exists():
        return {"status": "missing", "guard_skipped_at": None, "job_id": None}
    meta = json.loads(path.read_text())
    return {"status": meta.get("status"), "guard_skipped_at": meta.get("guard_skipped_at"),
            "job_id": meta.get("job_id")}


# ── Metric functions (tested in isolation) ────────────────────────────────────


def token_metrics(x: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-token cosine similarity and relative L2 error in float64.

    Args:
        x: Array of shape (T, H).
        ref: Reference array of the same shape.

    Returns:
        (cos, rel), each of shape (T,).

    Raises:
        ValueError: On shape mismatch, non-finite input or zero-norm rows.
    """
    x = np.asarray(x, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if x.ndim != 2 or x.shape != ref.shape:
        raise ValueError(f"shape mismatch {x.shape} vs {ref.shape}")
    if not (np.isfinite(x).all() and np.isfinite(ref).all()):
        raise ValueError("non-finite values")
    nr = np.linalg.norm(ref, axis=1)
    nx = np.linalg.norm(x, axis=1)
    if (nr == 0).any() or (nx == 0).any():
        raise ValueError("zero-norm token vector")
    cos = np.einsum("ij,ij->i", x, ref) / (nx * nr)
    rel = np.linalg.norm(x - ref, axis=1) / nr
    return cos, rel


def token_metrics_reference_impl(x: np.ndarray, ref: np.ndarray) -> tuple[list[float], list[float]]:
    """Independent pure-Python recomputation (math.fsum) of :func:`token_metrics`."""
    xs = np.asarray(x, dtype=np.float64).tolist()
    rs = np.asarray(ref, dtype=np.float64).tolist()
    cos_out: list[float] = []
    rel_out: list[float] = []
    for xv, rv in zip(xs, rs):
        dot = math.fsum(a * b for a, b in zip(xv, rv))
        nx = math.sqrt(math.fsum(a * a for a in xv))
        nr = math.sqrt(math.fsum(b * b for b in rv))
        nd = math.sqrt(math.fsum((a - b) * (a - b) for a, b in zip(xv, rv)))
        cos_out.append(dot / (nx * nr))
        rel_out.append(nd / nr)
    return cos_out, rel_out


def summarize_tokens(cos: np.ndarray, rel: np.ndarray) -> dict[str, float | int]:
    """Distribution summary over tokens (mean, median, p99, max; cosine low tail as p01/min)."""
    if cos.size == 0:
        return {"n_tokens": 0}
    dist = 1.0 - cos
    return {
        "n_tokens": int(cos.size),
        "cos_mean": float(cos.mean()), "cos_median": float(np.median(cos)),
        "cos_p01": float(np.percentile(cos, 1)), "cos_min": float(cos.min()),
        "cosdist_mean": float(dist.mean()), "cosdist_median": float(np.median(dist)),
        "cosdist_p99": float(np.percentile(dist, 99)), "cosdist_max": float(dist.max()),
        "rel_mean": float(rel.mean()), "rel_median": float(np.median(rel)),
        "rel_p99": float(np.percentile(rel, 99)), "rel_max": float(rel.max()),
    }


def log_softmax(v: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax in float64."""
    v = np.asarray(v, dtype=np.float64)
    m = v.max()
    return v - m - np.log(np.exp(v - m).sum())


def logits_metrics(logits: np.ndarray, ref_logits: np.ndarray, k: int) -> dict[str, Any]:
    """Compare last-token logits with the reference.

    Returns:
        top1_equal, top-k overlap count, KL(p_ref || p) in nats and the two top-1 ids.
    """
    if logits.shape != ref_logits.shape or logits.ndim != 1:
        raise ValueError(f"logits shape {logits.shape} vs reference {ref_logits.shape}")
    if not (np.isfinite(logits).all() and np.isfinite(ref_logits).all()):
        raise ValueError("non-finite logits")
    lp = log_softmax(logits)
    lr = log_softmax(ref_logits)
    pr = np.exp(lr)
    topk = set(np.argsort(-lp, kind="stable")[:k].tolist())
    topk_ref = set(np.argsort(-lr, kind="stable")[:k].tolist())
    top1 = int(np.argmax(lp))
    top1_ref = int(np.argmax(lr))
    return {"top1_equal": bool(top1 == top1_ref), f"top{k}_overlap": len(topk & topk_ref),
            "kl_ref_to_variant": float((pr * (lr - lp)).sum()), "top1": top1, "top1_ref": top1_ref}


def parse_speed_task(task_id: str) -> tuple[str, str, int]:
    """Split ``<model>__<format>__r<k>`` into (model, format, repeat)."""
    parts = task_id.split("__")
    if len(parts) != 3 or not parts[2].startswith("r") or not parts[2][1:].isdigit():
        raise ValueError(f"unexpected 0003 task id {task_id!r}")
    return parts[0], parts[1], int(parts[2][1:])


def bench_validity(result: dict[str, Any], num_prompts: int, output_len: int) -> tuple[bool, str]:
    """Apply the 0003 validity rule to one raw ``vllm bench serve`` result.

    Valid when no request failed, all ``num_prompts`` completed and every request
    produced exactly ``output_len`` tokens. Without per-request ``output_lens`` the
    total must equal ``completed * output_len``; since ``max_tokens = output_len`` caps
    each request, that equality implies every request has exactly ``output_len`` tokens.
    """
    for key in ("completed", "failed"):
        if key not in result:
            return False, f"missing key {key}"
    if int(result["failed"]) != 0:
        return False, f"{result['failed']} failed requests"
    if int(result["completed"]) != num_prompts:
        return False, f"completed {result['completed']} != {num_prompts}"
    lens = result.get("output_lens")
    if lens is not None:
        if len(lens) != num_prompts:
            return False, f"{len(lens)} output_lens for {num_prompts} prompts"
        bad = [n for n in lens if int(n) != output_len]
        if bad:
            return False, f"{len(bad)} requests with output length != {output_len}"
        return True, "ok (per-request output_lens)"
    total = result.get("total_output_tokens")
    if total is None:
        return False, "neither output_lens nor total_output_tokens present"
    if int(total) != num_prompts * output_len:
        return False, f"total_output_tokens {total} != {num_prompts * output_len}"
    return True, "ok (total_output_tokens; output_lens absent)"


def bench_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Recompute speed metrics of one valid raw ``vllm bench serve`` result.

    TPOT per request follows vLLM 0.31: ``(latency - ttft) / (output_len - 1)``; medians and
    p99 use ``np.percentile`` (linear), as in ``vllm/benchmarks/serve.py``.
    """
    duration = float(result["duration"])
    out: dict[str, Any] = {}
    detailed = all(k in result for k in ("ttfts", "latencies", "output_lens", "input_lens"))
    if detailed:
        ttfts = np.asarray(result["ttfts"], dtype=np.float64)
        lat = np.asarray(result["latencies"], dtype=np.float64)
        olen = np.asarray(result["output_lens"], dtype=np.float64)
        ilen = np.asarray(result["input_lens"], dtype=np.float64)
        if not (ttfts.size == lat.size == olen.size == ilen.size) or ttfts.size == 0:
            raise ValueError("per-request lists have inconsistent lengths")
        if (olen <= 1).any():
            raise ValueError("TPOT needs output_len > 1 for every request")
        tpot = (lat - ttfts) / (olen - 1)
        out.update({
            "output_throughput": float(olen.sum() / duration),
            "total_token_throughput": float((olen.sum() + ilen.sum()) / duration),
            "median_ttft_ms": float(np.median(ttfts) * 1000), "p99_ttft_ms": float(np.percentile(ttfts, 99) * 1000),
            "median_tpot_ms": float(np.median(tpot) * 1000), "p99_tpot_ms": float(np.percentile(tpot, 99) * 1000),
            "mean_input_len": float(ilen.mean()),
            "metric_source": "per_request",
        })
    else:
        for k in ("output_throughput", "total_token_throughput", "median_ttft_ms", "p99_ttft_ms",
                  "median_tpot_ms", "p99_tpot_ms"):
            out[k] = float(result[k]) if result.get(k) is not None else float("nan")
        n = int(result["completed"])
        out["mean_input_len"] = float(result["total_input_tokens"]) / n if n and "total_input_tokens" in result else float("nan")
        out["metric_source"] = "saved_summary"
    out["decode_tok_per_s"] = 1000.0 / out["median_tpot_ms"] if out["median_tpot_ms"] > 0 else float("nan")
    out["prefill_tok_per_s"] = (out["mean_input_len"] / (out["median_ttft_ms"] / 1000.0)
                                if out["median_ttft_ms"] > 0 else float("nan"))
    # Cross-check against the summary fields vLLM wrote into the same file.
    diffs = {}
    for k in ("output_throughput", "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms"):
        if result.get(k) is not None and out["metric_source"] == "per_request":
            diffs[k] = abs(out[k] - float(result[k])) / max(abs(float(result[k])), 1e-12)
    out["max_rel_diff_vs_saved_summary"] = max(diffs.values()) if diffs else float("nan")
    return out


def aggregate_repeats(values: list[float], min_valid: int) -> dict[str, Any]:
    """Median and min-max over valid repeats; ``incomplete`` below ``min_valid`` repeats."""
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    status = "complete" if len(vals) >= min_valid else "incomplete"
    if not vals:
        return {"n_valid": 0, "status": status, "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"n_valid": len(vals), "status": status, "median": float(np.median(vals)),
            "min": float(min(vals)), "max": float(max(vals))}


def layer_order(layer: str, n_layers: int) -> int:
    """Plot/sort position: emb = 0, pre_Lk = k, final_norm(_cb) = n_layers + 1."""
    if layer == "emb":
        return 0
    if layer.startswith("pre_L"):
        return int(layer[5:])
    if layer in ("final_norm", "final_norm_cb"):
        return n_layers + 1
    raise ValueError(f"unknown layer {layer}")


def norm_state(layer: str) -> str:
    """Normalization state of a layer name."""
    if layer == "emb":
        return "embedding"
    if layer.startswith("pre_L"):
        return "pre_norm"
    if layer in ("final_norm", "final_norm_cb"):
        return "normed"
    raise ValueError(f"unknown layer {layer}")


def reference_layer(layer: str, layer_map: dict[str, str]) -> str:
    """Reference layer to compare against (same normalization state by construction)."""
    ref = layer_map.get(layer, layer)
    if norm_state(ref) != norm_state(layer):
        raise ValueError(f"layer map pairs {layer} ({norm_state(layer)}) with {ref} ({norm_state(ref)})")
    return ref


# ── Speed ─────────────────────────────────────────────────────────────────────


def analyze_speed(project_root: Path, cfg: dict[str, Any], logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-repeat and per-cell speed tables from the raw 0003 files."""
    scfg = cfg["speed"]
    in_dir = project_root / scfg["input_dir"]
    workloads: dict[str, dict[str, Any]] = scfg["workloads"]
    out_len = int(scfg["output_len"])
    rep_rows: list[dict[str, Any]] = []
    found = sorted(p.name for p in in_dir.iterdir() if p.is_dir()) if in_dir.is_dir() else []
    expected = [f"{m}__{f}__r{r}" for m in scfg["models"] for f in scfg["formats"] for r in scfg["repeats"]]
    unexpected = sorted(set(found) - set(expected))
    if unexpected:
        logger.warning(f"0003 has unexpected task dirs (ignored, listed in checks): {unexpected}")

    for task in tqdm(expected, desc="0003 tasks"):
        model, fmt, rep = parse_speed_task(task)
        tdir = in_dir / task
        comp = read_completion(tdir)
        base = {"task_id": task, "model": model, "format": fmt, "engine": fmt.split("-")[0], "repeat": rep,
                "completion_status": comp["status"], "guard_skipped_at": comp["guard_skipped_at"],
                "upstream_job_id": comp["job_id"]}
        task_ok = comp["status"] == "completed"
        for wid, w in workloads.items():
            row = dict(base, workload=wid, kind="bench_serve")
            path = tdir / f"{wid}.json"
            if not task_ok:
                row.update(valid=False, reason=f"task completion status {comp['status']}")
            elif not path.exists():
                row.update(valid=False, reason="result file missing")
            else:
                result = json.loads(path.read_text())
                valid, reason = bench_validity(result, int(w["num_prompts"]), out_len)
                row.update(valid=valid, reason=reason, completed=result.get("completed"),
                           failed=result.get("failed"), max_concurrency_file=result.get("max_concurrency"))
                if result.get("max_concurrency") is not None and int(result["max_concurrency"]) != int(w["max_concurrency"]):
                    row.update(valid=False, reason=f"max_concurrency {result['max_concurrency']} != {w['max_concurrency']}")
                if row["valid"]:
                    row.update(bench_metrics(result))
            rep_rows.append(row)
        # Server-level metrics (one per task).
        row = dict(base, workload="server", kind="server")
        meta_path = tdir / "server_meta.json"
        if task_ok and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            row.update(valid=True, reason="ok", load_time_s=meta.get("load_time_s"), peak_vram_mib=meta.get("peak_vram_mib"))
        else:
            row.update(valid=False, reason="task not completed" if not task_ok else "server_meta.json missing")
        rep_rows.append(row)
        # llama-bench reference (llama.cpp only).
        if fmt.startswith("llamacpp"):
            row = dict(base, workload="llama_bench", kind="llama_bench")
            lb_path = tdir / "llama_bench.json"
            if task_ok and lb_path.exists():
                records = json.loads(lb_path.read_text())
                if not isinstance(records, list):
                    raise ValueError(f"{lb_path}: expected a JSON list of llama-bench records")
                row.update(valid=True, reason="ok")
                for name, (n_prompt, n_gen) in scfg["llama_bench"].items():
                    hits = [r for r in records if int(r.get("n_prompt", -1)) == n_prompt and int(r.get("n_gen", -1)) == n_gen]
                    if len(hits) > 1:
                        raise ValueError(f"{lb_path}: {len(hits)} records for {name}")
                    row[f"llama_bench_{name}_tok_per_s"] = float(hits[0]["avg_ts"]) if hits else float("nan")
            else:
                row.update(valid=False, reason="task not completed" if not task_ok else "llama_bench.json missing")
            rep_rows.append(row)
    reps = pd.DataFrame(rep_rows)

    metric_specs: list[tuple[str, str, str, list[str] | None]] = [
        ("bench_serve", "output_throughput", "tok/s", None),
        ("bench_serve", "total_token_throughput", "tok/s", None),
        ("bench_serve", "median_ttft_ms", "ms", scfg["ttft_workloads"]),
        ("bench_serve", "p99_ttft_ms", "ms", scfg["ttft_workloads"]),
        ("bench_serve", "prefill_tok_per_s", "tok/s", scfg["ttft_workloads"]),
        ("bench_serve", "median_tpot_ms", "ms", scfg["tpot_workloads"]),
        ("bench_serve", "p99_tpot_ms", "ms", scfg["tpot_workloads"]),
        ("bench_serve", "decode_tok_per_s", "tok/s", scfg["tpot_workloads"]),
        ("server", "load_time_s", "s", None),
        ("server", "peak_vram_mib", "MiB", None),
    ] + [("llama_bench", f"llama_bench_{n}_tok_per_s", "tok/s", None) for n in scfg["llama_bench"]]
    min_valid = int(scfg["min_valid_repeats"])
    cells: list[dict[str, Any]] = []
    for model in scfg["models"]:
        for fmt in scfg["formats"]:
            for kind, metric, unit, only in metric_specs:
                if kind == "llama_bench" and not fmt.startswith("llamacpp"):
                    continue
                wids = list(workloads) if kind == "bench_serve" else [kind]
                if only is not None:
                    wids = [w for w in wids if w in only]
                for wid in wids:
                    sub = reps[(reps.model == model) & (reps.format == fmt) & (reps.workload == wid)]
                    vals = sub.loc[sub.valid, metric].tolist() if metric in sub.columns else []
                    agg = aggregate_repeats(vals, min_valid)
                    cells.append({"model": model, "format": fmt, "engine": fmt.split("-")[0], "workload": wid,
                                  "metric": metric, "unit": unit, "n_repeats_expected": len(scfg["repeats"]),
                                  "n_repeats_valid_file": int(sub.valid.sum()), **agg,
                                  "values": json.dumps([float(v) for v in vals])})
    summary = pd.DataFrame(cells)
    base_fmt = scfg["baseline_format"]
    base = summary[summary.format == base_fmt].set_index(["model", "workload", "metric"])
    ratios = []
    for _, r in summary.iterrows():
        key = (r.model, r.workload, r.metric)
        b = base.loc[key] if key in base.index else None
        ok = b is not None and r.status == "complete" and b["status"] == "complete" and b["median"] != 0
        ratios.append(float(r["median"] / b["median"]) if ok else float("nan"))
    summary[f"ratio_to_{base_fmt}"] = ratios
    return reps, summary


# ── Fidelity ──────────────────────────────────────────────────────────────────


def load_prompt_ids(project_root: Path, cfg: dict[str, Any]) -> list[str]:
    """Prompt ids from the fixed 0004 prompt file (sha256 verified)."""
    hcfg = cfg["hidden"]
    path = project_root / hcfg["prompts_file"]
    digest = sha256_file(path)
    if digest != hcfg["prompts_sha256"]:
        raise ValueError(f"{path} sha256 {digest} != {hcfg['prompts_sha256']}")
    return [p["id"] for p in json.loads(path.read_text(encoding="utf-8"))["prompts"]]


def load_variant(in_dir: Path, variant: str) -> dict[str, Any]:
    """Completion status and capability table of one 0004 variant."""
    vdir = in_dir / variant
    comp = read_completion(vdir)
    cap_path = vdir / "capability.parquet"
    cap = pd.read_parquet(cap_path) if comp["status"] == "completed" and cap_path.exists() else None
    return {"dir": vdir, "completion": comp, "capability": cap}


def load_layer(vdir: Path, pid: str, layer: str) -> np.ndarray:
    """Load one saved array (float32 .npy)."""
    arr = np.load(vdir / pid / f"{layer}.npy", allow_pickle=False)
    if arr.dtype != np.float32:
        raise TypeError(f"{vdir / pid / layer}.npy dtype {arr.dtype}")
    return arr


def analyze_fidelity(
    project_root: Path, cfg: dict[str, Any], logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute fidelity_by_layer, logits_agreement, capability tables and check records."""
    hcfg = cfg["hidden"]
    in_dir = project_root / hcfg["input_dir"]
    n_layers = int(hcfg["num_hidden_layers"])
    hidden = int(hcfg["hidden_size"])
    ref_name = hcfg["reference"]
    k = int(hcfg["logits_top_k"])
    layer_map: dict[str, str] = hcfg["layer_reference_map"]
    prompt_ids = load_prompt_ids(project_root, cfg)
    variants = {v: load_variant(in_dir, v) for v in hcfg["variants"]}
    checks: dict[str, Any] = {"inputs_0004": {v: d["completion"] for v, d in variants.items()}}

    ref = variants[ref_name]
    if ref["capability"] is None:
        raise RuntimeError(f"reference {ref_name} is not completed; inputs too incomplete (consult)")
    ref_cap = ref["capability"]
    ref_ok = ref_cap.groupby("prompt_id").available.all()
    if set(ref_ok.index) != set(prompt_ids) or not ref_ok.all():
        raise RuntimeError(f"reference {ref_name} lacks arrays for some prompts: {ref_ok[~ref_ok].index.tolist()}")
    ref_layers = set(ref_cap.layer.unique())
    ref_ids = {pid: json.loads((ref["dir"] / pid / "token_ids.json").read_text())["token_ids"] for pid in prompt_ids}

    per_token: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    pooled: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    excluded: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    logit_rows: list[dict[str, Any]] = []
    cap_rows: list[dict[str, Any]] = []

    for vname, v in variants.items():
        cap = v["capability"]
        if cap is None:
            cap_rows.append({"variant": vname, "layer": None, "capable": False,
                             "completion_status": v["completion"]["status"],
                             "guard_skipped_at": v["completion"]["guard_skipped_at"],
                             "note": "variant not completed; unavailable"})
            continue
        extra_pre = {l for l in cap.layer.unique() if l.startswith("pre_L")} - {l for l in ref_layers if l.startswith("pre_L")}
        if extra_pre:
            raise RuntimeError(f"{vname} has layers {sorted(extra_pre)} absent from the reference: layer counts differ (consult)")
        ids_ok: dict[str, bool] = {}
        llama_tok: list[bool] = []
        for pid in prompt_ids:
            tpath = v["dir"] / pid / "token_ids.json"
            if tpath.exists():
                tinfo = json.loads(tpath.read_text())
                ids_ok[pid] = tinfo["token_ids"] == ref_ids[pid]
                if "llama_tokenization_equal_hf" in tinfo:
                    llama_tok.append(bool(tinfo["llama_tokenization_equal_hf"]))
            else:
                ids_ok[pid] = False
        for layer, lcap in cap.groupby("layer", sort=False):
            avail = lcap.set_index("prompt_id").available
            n_tok = lcap.set_index("prompt_id").n_tokens
            secs = lcap[lcap.available]
            if len(secs):
                scope = secs.timing_scope.iloc[0]
                denom = secs.seconds.sum() if scope == "prompt" else secs.seconds.drop_duplicates().sum()
                tps = float(secs.n_tokens.sum() / denom) if denom > 0 else float("nan")
            else:
                scope, tps = None, float("nan")
            capable = bool(len(avail) == len(prompt_ids) and avail.all() and all(ids_ok.values()))
            errors = sorted({str(e) for e in lcap.error.dropna().unique()})
            cap_rows.append({
                "variant": vname, "engine": lcap.engine.iloc[0], "method": lcap.method.iloc[0], "layer": layer,
                "norm_state": lcap.norm_state.iloc[0], "n_prompts": int(len(avail)), "n_available": int(avail.sum()),
                "n_token_ids_equal_ref": int(sum(ids_ok.values())), "capable": capable,
                "extraction_tok_per_s": tps, "timing_scope": scope,
                "n_tokens_available": int(n_tok[avail].sum()),
                "llama_tokenization_equal_hf_prompts": (int(sum(llama_tok)) if llama_tok else None),
                "errors": json.dumps(errors, ensure_ascii=False),
                "completion_status": v["completion"]["status"], "guard_skipped_at": v["completion"]["guard_skipped_at"],
                "note": None,
            })
        avail_map = {(r.prompt_id, r.layer): bool(r.available) for r in cap.itertuples()}
        layers = [l for l in cap.layer.unique() if l != "last_logits"]
        for layer in tqdm(layers, desc=f"fidelity {vname}", leave=False):
            rlayer = reference_layer(layer, layer_map)
            if rlayer not in ref_layers:
                raise RuntimeError(f"{vname}:{layer} has no reference layer {rlayer}")
            for pid in prompt_ids:
                if not avail_map.get((pid, layer), False):
                    excluded[(vname, layer)].append({"prompt_id": pid, "reason": "unavailable in 0004"})
                    continue
                if not ids_ok[pid]:
                    excluded[(vname, layer)].append({"prompt_id": pid, "reason": "token IDs differ from reference"})
                    continue
                x = load_layer(v["dir"], pid, layer)
                r = load_layer(ref["dir"], pid, rlayer)
                if x.shape != (len(ref_ids[pid]), hidden) or r.shape != x.shape:
                    excluded[(vname, layer)].append({"prompt_id": pid, "reason": f"shape {x.shape} vs ref {r.shape}"})
                    continue
                try:
                    cos, rel = token_metrics(x, r)
                except ValueError as e:
                    excluded[(vname, layer)].append({"prompt_id": pid, "reason": str(e)})
                    continue
                per_token[(vname, pid, layer)] = (cos, rel)
                pooled[(vname, layer)].append((cos, rel))
        # Last-token logits.
        if "last_logits" in set(cap.layer):
            for pid in prompt_ids:
                row = {"variant": vname, "level": "prompt", "prompt_id": pid}
                if not avail_map.get((pid, "last_logits"), False):
                    logit_rows.append(dict(row, available=False, reason="unavailable in 0004"))
                    continue
                if not ids_ok[pid]:
                    logit_rows.append(dict(row, available=False, reason="token IDs differ from reference"))
                    continue
                try:
                    m = logits_metrics(load_layer(v["dir"], pid, "last_logits"),
                                       load_layer(ref["dir"], pid, "last_logits"), k)
                    logit_rows.append(dict(row, available=True, reason=None, **m))
                except ValueError as e:
                    logit_rows.append(dict(row, available=False, reason=str(e)))

    rows: list[dict[str, Any]] = []
    for (vname, layer), lst in pooled.items():
        cos_all = np.concatenate([c for c, _ in lst])
        rel_all = np.concatenate([r for _, r in lst])
        cos_tail = np.concatenate([c[1:] for c, _ in lst])
        rel_tail = np.concatenate([r[1:] for _, r in lst])
        for scope, (c, r) in (("all", (cos_all, rel_all)), ("excl_first", (cos_tail, rel_tail))):
            rows.append({"variant": vname, "reference": ref_name, "layer": layer,
                         "reference_layer": reference_layer(layer, layer_map), "norm_state": norm_state(layer),
                         "layer_order": layer_order(layer, n_layers), "scope": scope, "n_prompts": len(lst),
                         **summarize_tokens(c, r),
                         "excluded_prompts": json.dumps(excluded.get((vname, layer), []), ensure_ascii=False)})
    for (vname, layer), exc in excluded.items():
        if (vname, layer) not in pooled:
            rows.append({"variant": vname, "reference": ref_name, "layer": layer,
                         "reference_layer": reference_layer(layer, layer_map), "norm_state": norm_state(layer),
                         "layer_order": layer_order(layer, n_layers), "scope": "all", "n_prompts": 0, "n_tokens": 0,
                         "excluded_prompts": json.dumps(exc, ensure_ascii=False)})
    fidelity = pd.DataFrame(rows).sort_values(["variant", "scope", "layer_order", "layer"]).reset_index(drop=True)

    # Sanity: reference vs itself.
    sanity = fidelity[(fidelity.variant == ref_name)]
    tol = float(cfg["sanity_cos_tolerance"])
    worst_cos = float(max(abs(sanity.cos_min - 1).max(), abs(sanity.cos_mean - 1).max()))
    worst_rel = float(sanity.rel_max.max())
    checks["sanity"] = {"reference": ref_name, "n_layers": int(sanity.layer.nunique()),
                        "max_abs_cos_minus_1": worst_cos, "max_rel": worst_rel, "tolerance_cos": tol}
    if sanity.empty or worst_cos > tol or worst_rel != 0.0:
        raise AssertionError(f"sanity check failed: {checks['sanity']}")

    # Independent recomputation on random (variant, prompt, layer) triples.
    keys = sorted(k for k in per_token if k[0] != ref_name)
    n_pairs = int(cfg["verification_n_pairs"])
    if len(keys) < n_pairs:
        raise RuntimeError(f"only {len(keys)} non-reference triples available for the independent check")
    rng = np.random.default_rng(int(cfg["verification_seed"]))
    picks = [keys[i] for i in rng.choice(len(keys), size=n_pairs, replace=False)]
    vt = float(cfg["verification_tolerance"])
    ver = []
    for vname, pid, layer in picks:
        x = load_layer(variants[vname]["dir"], pid, layer)
        r = load_layer(ref["dir"], pid, reference_layer(layer, layer_map))
        cos_i, rel_i = token_metrics_reference_impl(x, r)
        cos, rel = per_token[(vname, pid, layer)]
        dc = float(np.max(np.abs(np.asarray(cos_i) - cos)))
        dr = float(np.max(np.abs(np.asarray(rel_i) - rel)))
        ver.append({"variant": vname, "prompt_id": pid, "layer": layer, "max_abs_diff_cos": dc, "max_abs_diff_rel": dr})
    checks["independent_verification"] = {"seed": int(cfg["verification_seed"]), "tolerance": vt, "pairs": ver}
    if any(p["max_abs_diff_cos"] > vt or p["max_abs_diff_rel"] > vt for p in ver):
        raise AssertionError(f"independent recomputation disagrees: {ver}")

    logits = pd.DataFrame(logit_rows)
    if not logits.empty:
        means = []
        for vname, g in logits.groupby("variant", sort=False):
            ok = g[g.available]
            means.append({"variant": vname, "level": "mean", "prompt_id": None, "available": len(ok) > 0,
                          "n_prompts": int(len(ok)),
                          "top1_agreement_rate": float(ok.top1_equal.mean()) if len(ok) else float("nan"),
                          f"top{k}_overlap": float(ok[f"top{k}_overlap"].mean()) if len(ok) else float("nan"),
                          "kl_ref_to_variant": float(ok.kl_ref_to_variant.mean()) if len(ok) else float("nan"),
                          "kl_ref_to_variant_max": float(ok.kl_ref_to_variant.max()) if len(ok) else float("nan")})
        logits = pd.concat([logits, pd.DataFrame(means)], ignore_index=True)
    capability = pd.DataFrame(cap_rows)
    return fidelity, logits, capability, checks


# ── Figures ───────────────────────────────────────────────────────────────────


def save_figure(fig: Any, source: pd.DataFrame, fig_dir: Path, name: str, manifest: dict[str, Any]) -> None:
    """Save a figure (PNG) together with its source table."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    write_parquet(source.reset_index(drop=True), fig_dir / f"{name}.source.parquet")
    manifest["figures"][name] = {"png": f"{name}.png", "source": f"{name}.source.parquet", "rows": int(len(source))}


def make_figures(speed: pd.DataFrame, fidelity: pd.DataFrame, logits: pd.DataFrame, cfg: dict[str, Any],
                 fig_dir: Path, manifest: dict[str, Any]) -> None:
    """Draw the speed, fidelity and logits figures (matplotlib Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scfg = cfg["speed"]
    formats = scfg["formats"]
    models = scfg["models"]

    def bar_panel(ax: Any, sub: pd.DataFrame, xcats: list[str], title: str, ylabel: str) -> None:
        width = 0.8 / len(formats)
        for i, fmt in enumerate(formats):
            s = sub[sub.format == fmt].set_index("workload").reindex(xcats)
            ok = s.status == "complete"
            med = s["median"].where(ok)
            err = np.vstack([(med - s["min"]).fillna(0).to_numpy(dtype=float),
                             (s["max"] - med).fillna(0).to_numpy(dtype=float)])
            xs = np.arange(len(xcats)) + (i - (len(formats) - 1) / 2) * width
            ax.bar(xs, med.fillna(0).to_numpy(dtype=float), width, yerr=err, capsize=2, label=fmt)
            for x, flag in zip(xs.tolist(), ok.tolist()):
                if not flag:
                    ax.text(x, 0, "incomplete", rotation=90, ha="center", va="bottom", fontsize=6)
        ax.set_xticks(np.arange(len(xcats)), xcats, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel)

    # 1. Output throughput per workload.
    src = speed[speed.metric == "output_throughput"]
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models):
        bar_panel(ax, src[src.model == model], list(scfg["workloads"]), f"{model}: output throughput (median, min-max)",
                  "tok/s")
    axes[0][0].legend(fontsize=7)
    save_figure(fig, src, fig_dir, "speed_output_throughput", manifest)
    plt.close(fig)

    # 2. Latency: median TTFT (w1, w2) and median TPOT (w1).
    src = speed[speed.metric.isin(["median_ttft_ms", "median_tpot_ms"])]
    fig, axes = plt.subplots(2, len(models), figsize=(6 * len(models), 7), squeeze=False)
    for j, model in enumerate(models):
        bar_panel(axes[0][j], src[(src.model == model) & (src.metric == "median_ttft_ms")], scfg["ttft_workloads"],
                  f"{model}: median TTFT", "ms")
        bar_panel(axes[1][j], src[(src.model == model) & (src.metric == "median_tpot_ms")], scfg["tpot_workloads"],
                  f"{model}: median TPOT", "ms")
    axes[0][0].legend(fontsize=7)
    save_figure(fig, src, fig_dir, "speed_latency", manifest)
    plt.close(fig)

    # 3. Fidelity per layer vs hf-bf16 (excluding the first token), hf-fp32 dashed as precision floor.
    ref = cfg["hidden"]["reference"]
    floor = cfg["hidden"]["precision_floor"]
    src = fidelity[(fidelity.scope == "excl_first") & (fidelity.variant != ref) & (fidelity.n_tokens > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for vname, g in src.groupby("variant", sort=False):
        g = g.sort_values("layer_order")
        for layers, marker in ((g[g.layer != "final_norm_cb"], "o"), (g[g.layer == "final_norm_cb"], "x")):
            if layers.empty:
                continue
            style = "--" if vname == floor else "-"
            label = vname if marker == "o" else f"{vname} (final_norm_cb)"
            axes[0].plot(layers.layer_order, layers.cosdist_mean.clip(lower=1e-12), style, marker=marker, ms=3, label=label)
            axes[1].plot(layers.layer_order, layers.rel_mean.clip(lower=1e-12), style, marker=marker, ms=3, label=label)
    for ax, title in zip(axes, ["mean (1 - cosine) vs hf-bf16", "mean relative L2 error vs hf-bf16"]):
        ax.set_yscale("log")
        ax.set_xlabel("layer (0 = emb, k = after block k pre-norm, 37 = final_norm)")
        ax.set_title(title + " (tokens excl. position 0)", fontsize=9)
    axes[1].legend(fontsize=7)
    save_figure(fig, src, fig_dir, "fidelity_by_layer", manifest)
    plt.close(fig)

    # 4. Last-token logits agreement.
    if not logits.empty:
        src = logits[(logits.level == "mean") & (logits.variant != ref)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(src.variant, src.kl_ref_to_variant.clip(lower=1e-12))
        axes[0].set_yscale("log")
        axes[0].set_title("mean KL(p_hf-bf16 || p) at last prompt token", fontsize=9)
        axes[1].bar(src.variant, src.top1_agreement_rate)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("top-1 agreement rate with hf-bf16", fontsize=9)
        for ax in axes:
            ax.tick_params(axis="x", rotation=30, labelsize=8)
        save_figure(fig, src, fig_dir, "logits_agreement", manifest)
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")
    exp_dir = Path(__file__).resolve().parent
    cfg = load_config(exp_dir)
    run_dir = get_run_dir(project_root, __file__, VARIANT_KEY, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    rev = git_revision(project_root)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=VARIANT_KEY, git_commit=rev)
    logger.info(f"Starting: {exp_name} / {VARIANT_KEY}; run_dir {run_dir}")

    reps, speed = analyze_speed(project_root, cfg, logger)
    write_parquet(reps, run_dir / "speed_repeats.parquet")
    write_parquet(speed, run_dir / "speed_summary.parquet")
    logger.info(f"speed: {int(reps.valid.sum())}/{len(reps)} valid repeat rows; "
                f"{int((speed.status == 'incomplete').sum())}/{len(speed)} incomplete cells")

    fidelity, logits, capability, checks = analyze_fidelity(project_root, cfg, logger)
    write_parquet(fidelity, run_dir / "fidelity_by_layer.parquet")
    write_parquet(logits, run_dir / "logits_agreement.parquet")
    write_parquet(capability, run_dir / "capability.parquet")
    checks["inputs_0003"] = reps[["task_id", "completion_status", "guard_skipped_at"]].drop_duplicates().to_dict("records")
    checks["guard_skipped_inputs"] = (
        [t for t, c in checks["inputs_0004"].items() if c["guard_skipped_at"]]
        + sorted({r["task_id"] for r in checks["inputs_0003"] if r["guard_skipped_at"]})
    )
    (run_dir / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    logger.info(f"checks: sanity {checks['sanity']}; independent {checks['independent_verification']}")

    manifest: dict[str, Any] = {"experiment_py_sha256": sha256_file(Path(__file__)), "git_commit": rev, "figures": {}}
    make_figures(speed, fidelity, logits, cfg, run_dir / "figures", manifest)
    (run_dir / "figures" / "manifest.json").write_text(json.dumps(manifest, indent=1))

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
