"""0003 eval_engine_speed: serve one model/format with vLLM or llama.cpp and
measure it with ``vllm bench serve``.

One job = one plan task ``<model>__<engine_format>__r<repeat>``:

1. verify inputs (HF snapshot; GGUF SHA-256 for llama.cpp),
2. start the server, wait until healthy and record the load time,
3. for each of the 5 fixed workloads run ``vllm bench serve`` (4 warm-up
   requests via ``--num-warmups``, then the measured run) and save its JSON
   immediately,
4. sample ``nvidia-smi memory.used`` every 500 ms during the whole job,
5. stop the server and verify that it exited and released the GPU,
6. llama.cpp tasks only: run ``llama-bench`` once (after the server is gone,
   so the two processes never share the GPU),
7. write ``metrics.parquet`` (one row per workload) and ``server_meta.json``.

All fixed settings come from ``config.yml``; see plan.md for the approved
conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Output files (contract with 0005_analyze_engine_comparison).
SERVER_META_FILE = "server_meta.json"
VRAM_FILE = "vram.csv"
SERVER_LOG_FILE = "server.log"
LLAMA_BENCH_FILE = "llama_bench.json"
LLAMA_BENCH_LOG_FILE = "llama_bench.log"
METRICS_FILE = "metrics.parquet"
PACKAGES_FILE = "packages.txt"
EFFECTIVE_CONFIG_FILE = "effective_config.json"
PREVIOUS_ATTEMPTS_DIR = "previous_attempts"

# Column order of metrics.parquet (contract with 0005).
METRIC_COLUMNS: list[str] = [
    "model",
    "engine",
    "format",
    "repeat",
    "workload",
    "input_len",
    "output_len",
    "max_concurrency",
    "num_prompts",
    "completed",
    "failed",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "median_itl_ms",
    "mean_input_tokens_actual",
    "mean_output_tokens_actual",
    "valid",
    "invalid_reason",
    "load_time_s",
    "peak_vram_mib",
]

# Latency/throughput keys copied from the vllm bench serve result JSON.
BENCH_METRIC_KEYS: list[str] = [
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "median_itl_ms",
]

# Keys written into each result JSON with --metadata (identity for resume).
META_PREFIX = "meta_"


class TerminationRequested(RuntimeError):
    """Raised from a signal handler so that cleanup in ``finally`` runs."""


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (unit-tested without a GPU)
# ─────────────────────────────────────────────────────────────────────────────


def make_task_id(model: str, engine_format: str, repeat: int) -> str:
    """Build the plan task id used as variant_key.

    Args:
        model: Model key, e.g. ``qwen3-8b``.
        engine_format: Engine/format key, e.g. ``llamacpp-q8_0``.
        repeat: Repeat index (1-based).

    Returns:
        Task id such as ``qwen3-8b__llamacpp-q8_0__r2``.
    """
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    return f"{model}__{engine_format}__r{repeat}"


def workload_identity(task_id: str, engine: str, fmt: str, workload: dict[str, Any]) -> dict[str, str]:
    """Return the identity fields embedded in a result JSON via --metadata.

    Args:
        task_id: Plan task id.
        engine: ``vllm`` or ``llamacpp``.
        fmt: Format as written in the plan (e.g. ``bf16``, ``Q8_0``).
        workload: One workload entry from config.yml.

    Returns:
        Mapping of metadata key to string value (vllm stores them as strings).
    """
    return {
        f"{META_PREFIX}task_id": task_id,
        f"{META_PREFIX}engine": engine,
        f"{META_PREFIX}format": fmt,
        f"{META_PREFIX}workload": str(workload["id"]),
        f"{META_PREFIX}input_len": str(workload["input_len"]),
        f"{META_PREFIX}output_len": str(workload["output_len"]),
    }


def check_result_identity(
    result: dict[str, Any], identity: dict[str, str], workload: dict[str, Any]
) -> None:
    """Fail fast if a result JSON does not belong to this task/workload.

    Args:
        result: Parsed vllm bench serve result JSON.
        identity: Expected metadata from :func:`workload_identity`.
        workload: Workload entry from config.yml.

    Raises:
        ValueError: If any identity field, num_prompts or max_concurrency
            differs from the expected value.
    """
    for key, expected in identity.items():
        if str(result.get(key)) != expected:
            raise ValueError(f"result JSON {key}={result.get(key)!r} != expected {expected!r}")
    if result.get("num_prompts") != workload["num_prompts"]:
        raise ValueError(
            f"result JSON num_prompts={result.get('num_prompts')!r} != {workload['num_prompts']}"
        )
    if result.get("max_concurrency") != workload["max_concurrency"]:
        raise ValueError(
            f"result JSON max_concurrency={result.get('max_concurrency')!r} "
            f"!= {workload['max_concurrency']}"
        )


def validate_bench_result(
    result: dict[str, Any], num_prompts: int, output_len: int
) -> tuple[bool, str | None]:
    """Apply the plan validity rule to one vllm bench serve result.

    A workload is valid only when no request failed, all ``num_prompts``
    requests completed, and every completed request produced exactly
    ``output_len`` output tokens (plan evaluation.quality_checks.length_check).

    Args:
        result: Parsed result JSON saved with ``--save-detailed``.
        num_prompts: Number of measured requests of the workload.
        output_len: Required output tokens per request.

    Returns:
        ``(valid, invalid_reason)``; ``invalid_reason`` is None when valid and
        otherwise a ``;``-joined list of all detected problems.
    """
    reasons: list[str] = []
    missing = [k for k in ("completed", "failed", "output_lens", "errors") if k not in result]
    if missing:
        return False, f"missing_fields={','.join(missing)}"

    completed = int(result["completed"])
    failed = int(result["failed"])
    output_lens = list(result["output_lens"])
    errors = list(result["errors"])

    if failed != 0:
        reasons.append(f"failed_requests={failed}")
    if completed != num_prompts:
        reasons.append(f"completed={completed}!=num_prompts={num_prompts}")
    if len(output_lens) != num_prompts or len(errors) != num_prompts:
        reasons.append(
            f"per_request_count={len(output_lens)}/{len(errors)}!=num_prompts={num_prompts}"
        )

    success_lens = [n for n, err in zip(output_lens, errors) if not err]
    if len(success_lens) != completed:
        reasons.append(f"success_count_mismatch={len(success_lens)}!=completed={completed}")
    wrong = sorted({int(n) for n in success_lens if int(n) != output_len})
    n_wrong = sum(1 for n in success_lens if int(n) != output_len)
    if n_wrong:
        reasons.append(f"wrong_output_len_requests={n_wrong} (lens={wrong[:10]})")

    if reasons:
        return False, "; ".join(reasons)
    return True, None


def _as_float(value: Any) -> float:
    """Convert a JSON number (or None) to float, None -> NaN."""
    if value is None:
        return math.nan
    return float(value)


def build_metrics_row(
    result: dict[str, Any],
    workload: dict[str, Any],
    task: dict[str, Any],
    valid: bool,
    invalid_reason: str | None,
    load_time_s: float | None,
    peak_vram_mib: float | None,
) -> dict[str, Any]:
    """Build one metrics.parquet row from a vllm bench serve result.

    Args:
        result: Parsed result JSON.
        workload: Workload entry from config.yml.
        task: Dict with keys model, engine, format, repeat.
        valid: Output of :func:`validate_bench_result`.
        invalid_reason: Output of :func:`validate_bench_result`.
        load_time_s: Server load time of this task (None if unknown).
        peak_vram_mib: Peak GPU memory of the server session (None if unknown).

    Returns:
        Row dict with exactly the keys in ``METRIC_COLUMNS``.

    Raises:
        KeyError: If a valid result lacks one of the latency/throughput keys.
    """
    completed = int(result.get("completed", 0))
    failed = int(result.get("failed", 0))
    metrics: dict[str, float] = {}
    for key in BENCH_METRIC_KEYS:
        if key not in result:
            if valid:
                raise KeyError(f"valid result JSON lacks {key}")
            metrics[key] = math.nan
        else:
            metrics[key] = _as_float(result[key])

    if completed > 0:
        mean_in = float(result["total_input_tokens"]) / completed
        mean_out = float(result["total_output_tokens"]) / completed
    else:
        mean_in = math.nan
        mean_out = math.nan

    row: dict[str, Any] = {
        "model": task["model"],
        "engine": task["engine"],
        "format": task["format"],
        "repeat": int(task["repeat"]),
        "workload": workload["id"],
        "input_len": int(workload["input_len"]),
        "output_len": int(workload["output_len"]),
        "max_concurrency": int(workload["max_concurrency"]),
        "num_prompts": int(workload["num_prompts"]),
        "completed": completed,
        "failed": failed,
        **metrics,
        "mean_input_tokens_actual": mean_in,
        "mean_output_tokens_actual": mean_out,
        "valid": bool(valid),
        "invalid_reason": invalid_reason,
        "load_time_s": math.nan if load_time_s is None else float(load_time_s),
        "peak_vram_mib": math.nan if peak_vram_mib is None else float(peak_vram_mib),
    }
    assert list(row) == METRIC_COLUMNS, f"column mismatch: {list(row)}"
    return row


def load_valid_existing(
    json_path: Path,
    identity: dict[str, str],
    workload: dict[str, Any],
    output_len: int,
) -> dict[str, Any] | None:
    """Return a saved result if it exists, parses and passes the validity rule.

    Args:
        json_path: Path of ``<workload_id>.json``.
        identity: Expected metadata from :func:`workload_identity`.
        workload: Workload entry from config.yml.
        output_len: Required output tokens per request.

    Returns:
        The parsed result when it can be reused (resume skip), else None
        (missing, unparsable or invalid -> the workload is run again).

    Raises:
        ValueError: If the file parses but belongs to another task/workload
            (this indicates a configuration change and must not be resumed).
    """
    if not json_path.exists():
        return None
    try:
        result = json.loads(json_path.read_text())
    except json.JSONDecodeError:
        return None
    check_result_identity(result, identity, workload)
    valid, _ = validate_bench_result(result, int(workload["num_prompts"]), output_len)
    return result if valid else None


def build_client_cmd(
    client_cfg: dict[str, Any],
    seed: int,
    base_url: str,
    served_name: str,
    tokenizer: str,
    workload: dict[str, Any],
    result_dir: str,
    identity: dict[str, str],
) -> list[str]:
    """Build the ``vllm bench serve`` command for one workload.

    Flag names were checked against vllm 0.31.0
    (``vllm/benchmarks/serve.py`` and ``vllm/benchmarks/datasets/datasets.py``).

    Args:
        client_cfg: ``client`` section of config.yml.
        seed: Seed of the random dataset.
        base_url: Server base URL, e.g. ``http://127.0.0.1:18080``.
        served_name: Model name sent in requests.
        tokenizer: HF tokenizer path used to build the random prompts.
        workload: Workload entry from config.yml.
        result_dir: Directory where vllm writes the result JSON.
        identity: Metadata embedded into the result JSON.

    Returns:
        Command as an argv list.
    """
    cmd = [
        "vllm", "bench", "serve",
        "--backend", str(client_cfg["backend"]),
        "--base-url", base_url,
        "--endpoint", str(client_cfg["endpoint"]),
        "--model", served_name,
        "--tokenizer", tokenizer,
        "--dataset-name", str(client_cfg["dataset_name"]),
        "--random-input-len", str(workload["input_len"]),
        "--random-output-len", str(workload["output_len"]),
    ]
    if client_cfg["ignore_eos"]:
        cmd.append("--ignore-eos")
    cmd += [
        "--temperature", str(client_cfg["temperature"]),
        "--top-p", str(client_cfg["top_p"]),
        "--seed", str(seed),
        "--max-concurrency", str(workload["max_concurrency"]),
        "--num-prompts", str(workload["num_prompts"]),
        "--num-warmups", str(client_cfg["num_warmups"]),
        "--save-result",
    ]
    if client_cfg["save_detailed"]:
        cmd.append("--save-detailed")
    cmd += [
        "--result-dir", result_dir,
        "--result-filename", f"{workload['id']}.json",
        "--disable-tqdm",
        "--metadata",
        *[f"{k}={v}" for k, v in identity.items()],
    ]
    return cmd


def build_server_cmd(
    engine: str,
    fmt: str,
    config: dict[str, Any],
    model_cfg: dict[str, Any],
    gguf_path: Path | None,
    host: str,
    port: int,
) -> list[str]:
    """Build the server command for vLLM or llama-server.

    Args:
        engine: ``vllm`` or ``llamacpp``.
        fmt: Format key (``bf16``/``fp8`` for vLLM, ``BF16``/``Q8_0``/``Q4_K_M``).
        config: Full config.yml.
        model_cfg: Entry of ``config["models"]``.
        gguf_path: GGUF path (llama.cpp only).
        host: Bind host.
        port: Bind port.

    Returns:
        Command as an argv list.
    """
    if engine == "vllm":
        vcfg = config["vllm"]
        if fmt not in vcfg["format_args"]:
            raise ValueError(f"unknown vLLM format {fmt!r}")
        return [
            str(vcfg["executable"]), "serve", str(model_cfg["snapshot"]),
            "--served-model-name", str(model_cfg["hf_id"]),
            "--host", host, "--port", str(port),
            *[str(a) for a in vcfg["server_args"]],
            *[str(a) for a in vcfg["format_args"][fmt]],
        ]
    if engine == "llamacpp":
        if gguf_path is None:
            raise ValueError("llama.cpp server needs a GGUF path")
        lcfg = config["llamacpp"]
        return [
            str(lcfg["server_bin"]), "-m", str(gguf_path),
            *[str(a) for a in lcfg["server_args"]],
            "--host", host, "--port", str(port),
            "--alias", str(model_cfg["hf_id"]),
        ]
    raise ValueError(f"unknown engine {engine!r}")


def peak_in_window(samples: list[tuple[float, int]], start: float, end: float) -> int | None:
    """Return the maximum memory sample whose time lies in ``[start, end]``.

    Args:
        samples: ``(monotonic_time, memory_used_mib)`` pairs.
        start: Window start (monotonic seconds).
        end: Window end (monotonic seconds).

    Returns:
        Maximum MiB in the window, or None when no sample falls inside.
    """
    values = [mib for t, mib in samples if start <= t <= end]
    return max(values) if values else None


def sha256_file(path: Path, chunk_size: int = 64 * 1024 * 1024) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: File to hash.
        chunk_size: Read size in bytes.

    Returns:
        Lower-case hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_gguf(gguf_path: Path, meta_filename: str, sha_key: str) -> str:
    """Verify a GGUF against the SHA-256 written by 0001/0002.

    Args:
        gguf_path: GGUF file.
        meta_filename: Name of the metadata JSON next to the GGUF.
        sha_key: Key holding the expected SHA-256.

    Returns:
        The verified SHA-256.

    Raises:
        FileNotFoundError: If the GGUF, its metadata or the upstream
            completion.json is missing.
        ValueError: If the upstream run is not completed, the metadata lacks
            the key, or the digest does not match.
    """
    if not gguf_path.is_file():
        raise FileNotFoundError(f"GGUF not found: {gguf_path}")
    completion = gguf_path.parent / "completion.json"
    if not completion.is_file():
        raise FileNotFoundError(f"upstream completion.json not found: {completion}")
    status = json.loads(completion.read_text()).get("status")
    if status != "completed":
        raise ValueError(f"upstream run not completed ({status!r}): {completion}")
    meta_path = gguf_path.parent / meta_filename
    if not meta_path.is_file():
        raise FileNotFoundError(f"GGUF metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text())
    if "gguf_file" in meta and meta["gguf_file"] != gguf_path.name:
        raise ValueError(f"{meta_path} describes {meta['gguf_file']!r}, not {gguf_path.name!r}")
    expected = meta.get(sha_key)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{meta_path} has no valid {sha_key!r}: {expected!r}")
    actual = sha256_file(gguf_path)
    if actual != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {gguf_path}: {actual} != {expected}")
    return actual


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically (tmp file + rename)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically."""
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_metrics_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write metrics rows to parquet atomically, with fixed column order.

    Args:
        path: Destination ``metrics.parquet``.
        rows: Rows from :func:`build_metrics_row`.
    """
    import pandas as pd

    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    df["invalid_reason"] = df["invalid_reason"].astype("string")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
# Process / GPU helpers (need the compute node)
# ─────────────────────────────────────────────────────────────────────────────


def query_gpu_memory_used() -> int:
    """Return ``memory.used`` [MiB] of the single visible GPU.

    Raises:
        RuntimeError: If nvidia-smi fails or reports a GPU count other than 1.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout.strip().splitlines()
    if len(out) != 1:
        raise RuntimeError(f"expected exactly 1 visible GPU, nvidia-smi reported {len(out)}")
    return int(out[0].strip())


def query_gpu_info() -> dict[str, str]:
    """Return name, driver version and total memory of the visible GPU."""
    line = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout.strip()
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        raise RuntimeError(f"unexpected nvidia-smi output: {line!r}")
    return {"gpu_name": parts[0], "driver_version": parts[1], "memory_total": parts[2]}


def wait_gpu_below(threshold_mib: int, timeout_s: float) -> tuple[bool, int]:
    """Wait until GPU memory used drops below a threshold.

    Args:
        threshold_mib: Threshold in MiB.
        timeout_s: Maximum wait in seconds.

    Returns:
        ``(reached, last_value_mib)``.
    """
    deadline = time.monotonic() + timeout_s
    used = query_gpu_memory_used()
    while used >= threshold_mib and time.monotonic() < deadline:
        time.sleep(1.0)
        used = query_gpu_memory_used()
    return used < threshold_mib, used


class VramSampler:
    """Stream ``nvidia-smi memory.used`` every N ms into ``vram.csv``.

    Each sample is written and flushed immediately (timestamp in local ISO
    format, memory in MiB) and kept in memory with its monotonic receive time
    so that peaks can be computed for a time window.
    """

    def __init__(self, csv_path: Path, interval_ms: int) -> None:
        self.csv_path = csv_path
        self.interval_ms = interval_ms
        self.samples: list[tuple[float, int]] = []
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start nvidia-smi in loop mode and the reader thread."""
        new_file = not self.csv_path.exists()
        self._fh = open(self.csv_path, "a", buffering=1)
        if new_file:
            self._fh.write("timestamp,memory_used_mib\n")
        self._proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-lms", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                mib = int(line)
            except ValueError:
                continue
            now_mono = time.monotonic()
            with self._lock:
                self.samples.append((now_mono, mib))
            self._fh.write(f"{datetime.now().isoformat()},{mib}\n")

    def snapshot(self) -> list[tuple[float, int]]:
        """Return a copy of the samples collected so far."""
        with self._lock:
            return list(self.samples)

    def stop(self) -> None:
        """Stop nvidia-smi and close the CSV (safe to call twice)."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if getattr(self, "_fh", None) is not None and not self._fh.closed:
            self._fh.close()


def port_is_free(host: str, port: int) -> bool:
    """Return True when nothing accepts connections on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) != 0


@dataclass
class ManagedServer:
    """A server process started in its own session (process group)."""

    cmd: list[str]
    log_path: Path
    env: dict[str, str]
    proc: subprocess.Popen[bytes] | None = None
    t_start: float = 0.0
    _log_fh: Any = field(default=None, repr=False)

    def start(self) -> None:
        """Start the server; stdout/stderr go to ``server.log``."""
        self._log_fh = open(self.log_path, "ab")
        self.t_start = time.monotonic()
        self.proc = subprocess.Popen(
            self.cmd, stdout=self._log_fh, stderr=subprocess.STDOUT,
            env=self.env, start_new_session=True,
        )

    def alive(self) -> bool:
        """Return True while the main server process runs."""
        return self.proc is not None and self.proc.poll() is None

    def wait_healthy(self, url: str, require_ok_status: bool, timeout_s: float,
                     poll_s: float) -> float:
        """Poll the health endpoint until HTTP 200 and return the load time.

        Args:
            url: Health URL.
            require_ok_status: llama-server: also require ``{"status": "ok"}``.
            timeout_s: Maximum wait.
            poll_s: Poll interval.

        Returns:
            Seconds from process start to the first healthy response.

        Raises:
            RuntimeError: If the process exits or the timeout is reached.
        """
        while True:
            if not self.alive():
                raise RuntimeError(f"server exited before healthy (code {self.proc.returncode})")
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    body = resp.read()
                    if resp.status == 200:
                        if not require_ok_status:
                            return time.monotonic() - self.t_start
                        if json.loads(body or b"{}").get("status") == "ok":
                            return time.monotonic() - self.t_start
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError,
                    json.JSONDecodeError):
                pass
            if time.monotonic() - self.t_start > timeout_s:
                raise RuntimeError(f"server not healthy after {timeout_s}s")
            time.sleep(poll_s)

    def stop(self, timeout_s: float) -> bool:
        """SIGTERM the process group, SIGKILL after timeout; verify exit.

        Args:
            timeout_s: Grace period before SIGKILL.

        Returns:
            True when the main process exited and no process of its group
            remains.
        """
        if self.proc is None:
            return True
        pgid = self.proc.pid
        if self.proc.poll() is None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                pass
        # Children (e.g. vLLM EngineCore) may outlive the main process.
        deadline = time.monotonic() + timeout_s
        while _group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.5)
        if _group_alive(pgid) or self.proc.poll() is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            time.sleep(1.0)
        if self._log_fh is not None and not self._log_fh.closed:
            self._log_fh.close()
        return self.proc.poll() is not None and not _group_alive(pgid)


def _group_alive(pgid: int) -> bool:
    """Return True if any process of the process group still exists."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_child(cmd: list[str], env: dict[str, str], log_path: Path, timeout_s: float) -> int:
    """Run a child process with output appended to a log; kill it on errors.

    Args:
        cmd: Command argv.
        env: Environment.
        log_path: Log file (stdout and stderr are appended).
        timeout_s: Timeout in seconds.

    Returns:
        Exit code.

    Raises:
        RuntimeError: On timeout.
    """
    with open(log_path, "ab") as log:
        log.write(f"\n$ {shlex.join(cmd)}\n".encode())
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"timeout after {timeout_s}s: {shlex.join(cmd)}") from e
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def llamacpp_version(server_bin: str) -> str:
    """Return the ``llama-server --version`` text."""
    res = subprocess.run([server_bin, "--version"], capture_output=True, text=True, timeout=120)
    text = (res.stdout + res.stderr).strip()
    if res.returncode != 0 or not text:
        raise RuntimeError(f"{server_bin} --version failed ({res.returncode}): {text}")
    return text


def git_revision(project_root: Path) -> str | None:
    """Return the repository HEAD revision, or None if git is unavailable."""
    try:
        res = subprocess.run(["git", "-C", str(project_root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def write_package_list(path: Path) -> None:
    """Save installed distributions (uv pip freeze equivalent)."""
    pkgs = sorted(
        {f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
         if d.metadata["Name"]},
        key=str.lower,
    )
    atomic_write_text(path, "\n".join(pkgs) + "\n")


def archive_previous_attempt(run_dir: Path, names: list[str]) -> Path | None:
    """Move files of an earlier attempt into ``previous_attempts/<ts>/``.

    Raw files are kept (plan outputs.preservation), never overwritten.

    Args:
        run_dir: Run directory.
        names: File names to move if present.

    Returns:
        The archive directory, or None when nothing was moved.
    """
    present = [n for n in names if (run_dir / n).exists()]
    if not present:
        return None
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    dest = run_dir / PREVIOUS_ATTEMPTS_DIR / stamp
    suffix = 0
    while dest.exists():
        suffix += 1
        dest = run_dir / PREVIOUS_ATTEMPTS_DIR / f"{stamp}_{suffix}"
    dest.mkdir(parents=True)
    for n in present:
        shutil.move(str(run_dir / n), str(dest / n))
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Template plumbing
# ─────────────────────────────────────────────────────────────────────────────


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
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yml not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("config.yml must be a mapping")
    return config


def parse_args(config: dict[str, Any], argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the swept dimensions (must match GRID_ARGS in run_slurm.sh).

    Args:
        config: Loaded config.yml (provides the allowed values).
        argv: Optional argv for tests.

    Returns:
        Namespace with model, engine_format and repeat.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(config["models"]))
    parser.add_argument("--engine_format", required=True, choices=sorted(config["engine_formats"]))
    parser.add_argument("--repeat", required=True, type=int, choices=list(config["repeats"]))
    return parser.parse_args(argv)


def _install_signal_handlers(logger: logging.Logger) -> None:
    """Turn SIGTERM/SIGINT/SIGUSR1 into an exception so cleanup runs."""

    def handler(signum: int, _frame: Any) -> None:
        # Ignore further signals so they cannot interrupt the cleanup.
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(s, signal.SIG_IGN)
        logger.error(f"received signal {signal.Signals(signum).name}; stopping")
        raise TerminationRequested(f"signal {signal.Signals(signum).name}")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, handler)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from tqdm import tqdm

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    config = load_config(Path(__file__).parent)
    args = parse_args(config)

    ef = config["engine_formats"][args.engine_format]
    engine: str = ef["engine"]
    fmt: str = ef["format"]
    model_cfg: dict[str, Any] = config["models"][args.model]
    task_id = make_task_id(args.model, args.engine_format, args.repeat)
    task = {"model": args.model, "engine": engine, "format": fmt, "repeat": args.repeat}
    output_len = int(config["output_len"])
    workloads: list[dict[str, Any]] = config["workloads"]
    for wl in workloads:
        if int(wl["output_len"]) != output_len:
            raise ValueError(f"workload {wl['id']} output_len != {output_len}")

    variant_key = task_id
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    canonical_dir = project_root / "outputs" / exp_name / variant_key
    logger = setup_logger(run_dir, exp_name)
    _install_signal_handlers(logger)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, **task)
    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir: {run_dir} (canonical: {canonical_dir})")

    # Resume: bring valid workload JSONs of an earlier attempt into run_dir
    # when run_dir is a fresh scratch directory.
    if run_dir.resolve() != canonical_dir.resolve() and canonical_dir.is_dir():
        for wl in workloads:
            src = canonical_dir / f"{wl['id']}.json"
            if src.exists() and not (run_dir / src.name).exists():
                shutil.copy2(src, run_dir / src.name)
    archived = archive_previous_attempt(
        run_dir, [SERVER_META_FILE, VRAM_FILE, SERVER_LOG_FILE, LLAMA_BENCH_FILE,
                  LLAMA_BENCH_LOG_FILE, METRICS_FILE, "client.log"],
    )
    if archived is not None:
        logger.info(f"archived files of an earlier attempt to {archived}")

    # ── Inputs ────────────────────────────────────────────────────────────────
    snapshot = Path(model_cfg["snapshot"])
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"HF snapshot incomplete: {snapshot}")
    gguf_path: Path | None = None
    gguf_sha: str | None = None
    if engine == "llamacpp":
        gguf_path = project_root / model_cfg["gguf"][fmt]
        logger.info(f"verifying SHA-256 of {gguf_path}")
        gguf_sha = verify_gguf(gguf_path, config["gguf_meta_filename"], config["gguf_meta_sha256_key"])
        logger.info(f"GGUF SHA-256 OK: {gguf_sha}")

    host = str(config["server"]["host"])
    port = int(config["server"]["port"])
    base_url = f"http://{host}:{port}"
    server_cmd = build_server_cmd(engine, fmt, config, model_cfg, gguf_path, host, port)
    if engine == "vllm":
        resolved = shutil.which(server_cmd[0])
        if resolved is None:
            raise FileNotFoundError(f"{server_cmd[0]} not on PATH (is .venv active?)")
        engine_version = importlib.metadata.version("vllm")
    else:
        if not Path(server_cmd[0]).is_file():
            raise FileNotFoundError(server_cmd[0])
        engine_version = llamacpp_version(server_cmd[0])
    client_template = build_client_cmd(
        config["client"], int(config["seed"]), base_url, str(model_cfg["hf_id"]), str(snapshot),
        {"id": "{workload_id}", "input_len": "{input_len}", "output_len": "{output_len}",
         "max_concurrency": "{max_concurrency}", "num_prompts": "{num_prompts}"},
        "{run_dir}", {"meta_...": "..."},
    )

    base_env = dict(os.environ)
    base_env["HF_HUB_OFFLINE"] = "1"
    base_env["TRANSFORMERS_OFFLINE"] = "1"
    client_env = dict(base_env)
    # The client never needs the GPU; hide it so it cannot add GPU memory.
    client_env["CUDA_VISIBLE_DEVICES"] = ""

    meta: dict[str, Any] = {
        "task_id": task_id,
        **task,
        "hf_id": model_cfg["hf_id"],
        "hf_snapshot": str(snapshot),
        "load_time_s": None,
        "peak_vram_mib": None,
        "peak_vram_mib_job": None,
        "baseline_vram_mib": None,
        "server_cmd": shlex.join(server_cmd),
        "client_cmd_template": shlex.join(client_template),
        "engine_version": engine_version,
        "client_vllm_version": importlib.metadata.version("vllm"),
        "gguf_path": None if gguf_path is None else str(gguf_path),
        "gguf_sha256": gguf_sha,
        "workload_ids": [wl["id"] for wl in workloads],
        "resumed_workloads": [],
        "server_exit_verified": None,
        "git_revision": git_revision(project_root),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "host": socket.gethostname(),
        **query_gpu_info(),
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "status": "running",
        "error": None,
    }
    meta_path = run_dir / SERVER_META_FILE
    atomic_write_json(meta_path, meta)
    atomic_write_json(run_dir / EFFECTIVE_CONFIG_FILE, {"task": task, "task_id": task_id, "config": config})
    write_package_list(run_dir / PACKAGES_FILE)

    # ── Measurement ───────────────────────────────────────────────────────────
    gcfg = config["gpu"]
    scfg = config["server"]
    sampler = VramSampler(run_dir / VRAM_FILE, int(gcfg["sample_interval_ms"]))
    server: ManagedServer | None = None
    rows: list[dict[str, Any]] = []
    t_server_start = t_server_stop = None
    metrics_path = run_dir / METRICS_FILE
    try:
        if not port_is_free(host, port):
            raise RuntimeError(f"port {host}:{port} is already in use")
        free, used = wait_gpu_below(int(gcfg["free_threshold_mib"]), float(gcfg["free_timeout_s"]))
        if not free:
            raise RuntimeError(f"GPU not free before start: {used} MiB used")
        meta["baseline_vram_mib"] = used
        sampler.start()

        server = ManagedServer(server_cmd, run_dir / SERVER_LOG_FILE, base_env)
        logger.info(f"starting server: {shlex.join(server_cmd)}")
        server.start()
        t_server_start = server.t_start
        load_time = server.wait_healthy(
            f"{base_url}/health", require_ok_status=(engine == "llamacpp"),
            timeout_s=float(scfg["health_timeout_s"]), poll_s=float(scfg["health_poll_interval_s"]),
        )
        meta["load_time_s"] = load_time
        atomic_write_json(meta_path, meta)
        logger.info(f"server healthy after {load_time:.2f}s")

        tmp_root = run_dir / ".bench_tmp"
        for wl in tqdm(workloads, desc=task_id, unit="workload"):
            identity = workload_identity(task_id, engine, fmt, wl)
            json_path = run_dir / f"{wl['id']}.json"
            result = load_valid_existing(json_path, identity, wl, output_len)
            if result is not None:
                logger.info(f"[{wl['id']}] valid result exists; skipped (resume)")
                meta["resumed_workloads"].append(wl["id"])
            else:
                if json_path.exists():
                    archive_previous_attempt(run_dir, [json_path.name])
                tmp_dir = tmp_root / str(wl["id"])
                tmp_dir.mkdir(parents=True, exist_ok=True)
                cmd = build_client_cmd(
                    config["client"], int(config["seed"]), base_url, str(model_cfg["hf_id"]),
                    str(snapshot), wl, str(tmp_dir), identity,
                )
                logger.info(f"[{wl['id']}] {shlex.join(cmd)}")
                code = run_child(cmd, client_env, run_dir / "client.log",
                                 float(config["client"]["timeout_s"]))
                tmp_json = tmp_dir / f"{wl['id']}.json"
                if code != 0 or not tmp_json.is_file():
                    raise RuntimeError(f"[{wl['id']}] vllm bench serve failed (exit {code})")
                os.replace(tmp_json, json_path)
                result = json.loads(json_path.read_text())
                check_result_identity(result, identity, wl)
            valid, reason = validate_bench_result(result, int(wl["num_prompts"]), output_len)
            peak_so_far = peak_in_window(sampler.snapshot(), t_server_start, time.monotonic())
            rows.append(build_metrics_row(result, wl, task, valid, reason, load_time, peak_so_far))
            write_metrics_parquet(metrics_path, rows)
            logger.info(
                f"[{wl['id']}] valid={valid} reason={reason} "
                f"out_tok/s={result.get('output_throughput')}"
            )
            if not server.alive():
                raise RuntimeError(f"server died during {wl['id']} (code {server.proc.returncode})")
        shutil.rmtree(tmp_root, ignore_errors=True)

        # Stop the server and verify that it is gone and the GPU is released.
        t_server_stop = time.monotonic()
        exited = server.stop(float(scfg["stop_timeout_s"]))
        released, used_after = wait_gpu_below(
            int(meta["baseline_vram_mib"]) + int(gcfg["free_threshold_mib"]),
            float(gcfg["free_timeout_s"]),
        )
        meta["server_exit_verified"] = bool(exited and released)
        if not meta["server_exit_verified"]:
            raise RuntimeError(
                f"server shutdown not verified (exited={exited}, gpu_used_after={used_after} MiB)"
            )
        logger.info("server stopped and GPU released")

        peak = peak_in_window(sampler.snapshot(), t_server_start, t_server_stop)
        meta["peak_vram_mib"] = peak
        rows = [{**r, "peak_vram_mib": math.nan if peak is None else float(peak)} for r in rows]
        write_metrics_parquet(metrics_path, rows)
        atomic_write_json(meta_path, meta)

        # llama-bench reference (llama.cpp only), after the server is gone.
        if engine == "llamacpp":
            assert gguf_path is not None
            lcfg = config["llamacpp"]
            bench_cmd = [str(lcfg["bench_bin"]), "-m", str(gguf_path),
                         *[str(a) for a in lcfg["bench_args"]]]
            meta["llama_bench_cmd"] = shlex.join(bench_cmd)
            logger.info(f"llama-bench: {meta['llama_bench_cmd']}")
            tmp = run_dir / f".{LLAMA_BENCH_FILE}.tmp"
            with open(tmp, "wb") as out, open(run_dir / LLAMA_BENCH_LOG_FILE, "ab") as err:
                proc = subprocess.Popen(bench_cmd, stdout=out, stderr=err, env=base_env)
                try:
                    code = proc.wait(timeout=float(config["llama_bench_timeout_s"]))
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
            if code != 0:
                raise RuntimeError(f"llama-bench failed (exit {code}); see {LLAMA_BENCH_LOG_FILE}")
            parsed = json.loads(tmp.read_text())
            if not isinstance(parsed, list) or not parsed:
                raise RuntimeError("llama-bench JSON is not a non-empty list")
            os.replace(tmp, run_dir / LLAMA_BENCH_FILE)

        meta["status"] = "completed"
    except BaseException as e:
        meta["status"] = "failed"
        meta["error"] = f"{type(e).__name__}: {e}"
        logger.exception("task failed")
        raise
    finally:
        if server is not None and server.alive():
            ok = server.stop(float(scfg["stop_timeout_s"]))
            if meta["server_exit_verified"] is None:
                meta["server_exit_verified"] = ok
        sampler.stop()
        meta["peak_vram_mib_job"] = max((m for _, m in sampler.snapshot()), default=None)
        if meta["peak_vram_mib"] is None and t_server_start is not None:
            meta["peak_vram_mib"] = peak_in_window(
                sampler.snapshot(), t_server_start, t_server_stop or time.monotonic()
            )
        meta["finished_at"] = datetime.now().isoformat()
        atomic_write_json(meta_path, meta)
        if rows:
            write_metrics_parquet(metrics_path, rows)

    complete_run(run_dir)
    n_valid = sum(r["valid"] for r in rows)
    logger.info(f"Done: {n_valid}/{len(rows)} workloads valid")


if __name__ == "__main__":
    main()
