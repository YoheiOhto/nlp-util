"""0002 preprocess_gguf_quantize: quantize one 0001 BF16 GGUF to one llama.cpp type.

One job handles one (model, type) pair; variant_key == plan task id
(e.g. ``qwen3-8b__q8_0``):

1. Verify the runtime (llama.cpp ref/commit from the SIF labels).
2. Verify the 0001 input: completed, load check passed, and the recomputed
   SHA-256 / size equal the values recorded in 0001 ``gguf_meta.json``.
3. ``llama-quantize <bf16> model-<TYPE>.gguf.tmp <TYPE> 16`` (no imatrix).
4. Rename ``.tmp`` after exit 0, record SHA-256 / size.
5. Load check with llama.cpp gguf-py (tensor count equals the input, expected
   ``general.file_type`` and tensor type present).
6. Write ``gguf_meta.json`` and ``results.parquet``.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import resource
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

# Inline reader executed with the llama.cpp venv python (the gguf package is not
# in the orchestration venv). argv[1] = gguf-py source dir, argv[2] = GGUF path.
GGUF_INSPECT_SCRIPT = r"""
import json, os, sys
from collections import Counter
sys.path.insert(0, sys.argv[1])
import gguf
from gguf import GGUFReader
path = sys.argv[2]
reader = GGUFReader(path, "r")
def scalar(key):
    field = reader.get_field(key)
    return None if field is None else field.contents()
counts = Counter(t.tensor_type.name for t in reader.tensors)
data_end = max((int(t.data_offset) + int(t.n_bytes) for t in reader.tensors), default=0)
print(json.dumps({
    "tensor_count": len(reader.tensors),
    "tensor_type_counts": dict(sorted(counts.items())),
    "file_type": scalar("general.file_type"),
    "architecture": scalar("general.architecture"),
    "kv_count": len(reader.fields),
    "data_end": data_end,
    "file_size": os.path.getsize(path),
    "gguf_module": gguf.__file__,
}))
"""

SIZE_RE = {
    "model_size_mib": re.compile(r"model size\s*=\s*([\d.]+)\s*MiB"),
    "quant_size_mib": re.compile(r"quant size\s*=\s*([\d.]+)\s*MiB"),
}


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


def load_config(exp_dir: Path) -> dict:
    """Load config.yml from the experiment directory.

    Raises:
        FileNotFoundError: If config.yml is missing.
    """
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yml not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the swept dimensions (must match GRID_ARGS in run_slurm.sh).

    Args:
        argv: Argument list (None uses sys.argv).

    Returns:
        Parsed arguments with ``model`` and ``type``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model id (0001 variant id).")
    parser.add_argument("--type", required=True, help="llama-quantize type, e.g. Q8_0 or Q4_K_M.")
    return parser.parse_args(argv)


def validate_args(config: dict, model: str, qtype: str) -> dict:
    """Check that the pair is one of the approved tasks and return the type spec.

    Raises:
        ValueError: If the model or type is not in config.yml.
    """
    if model not in config["models"]:
        raise ValueError(f"Unknown --model {model!r}; allowed: {config['models']}")
    if qtype not in config["quant_types"]:
        raise ValueError(f"Unknown --type {qtype!r}; allowed: {sorted(config['quant_types'])}")
    return config["quant_types"][qtype]


def build_variant_key(model: str, qtype: str) -> str:
    """Return the variant key; equals the plan task id (e.g. ``qwen3-8b__q4_k_m``)."""
    return f"{model}__{qtype.lower()}"


def read_llamacpp_labels(labels_file: Path, expected_ref: str, expected_commit: str) -> dict[str, str]:
    """Read the SIF labels and verify the llama.cpp ref/commit.

    Raises:
        FileNotFoundError: If the labels file is missing (not inside the SIF).
        ValueError: If the ref or commit differs from the approved runtime.
    """
    if not labels_file.exists():
        raise FileNotFoundError(f"SIF labels not found: {labels_file} (not running inside llamacpp.sif?)")
    labels = json.loads(labels_file.read_text())
    ref = labels.get("org.nlp-util.llamacpp-ref")
    commit = labels.get("org.nlp-util.llamacpp-commit")
    if ref != expected_ref or commit != expected_commit:
        raise ValueError(
            f"llama.cpp runtime mismatch: ref={ref!r} commit={commit!r}; "
            f"expected ref={expected_ref!r} commit={expected_commit!r}"
        )
    return {"llamacpp_ref": ref, "llamacpp_commit": commit}


def sha256_file(path: Path, chunk_bytes: int) -> str:
    """Stream a file through SHA-256 with a tqdm progress bar."""
    digest = hashlib.sha256()
    total = path.stat().st_size
    with open(path, "rb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=f"sha256 {path.name}", mininterval=30, file=sys.stderr
    ) as bar:
        while chunk := f.read(chunk_bytes):
            digest.update(chunk)
            bar.update(len(chunk))
    return digest.hexdigest()


def load_upstream_meta(upstream_dir: Path, meta_filename: str, gguf_filename: str) -> dict[str, Any]:
    """Load the 0001 metadata and check that the variant finished and passed its load check.

    Args:
        upstream_dir: ``outputs/<0001>/<model>`` directory.
        meta_filename: Name of the 0001 metadata file (``gguf_meta.json``).
        gguf_filename: Expected GGUF filename (``model-BF16.gguf``).

    Returns:
        The 0001 metadata dict.

    Raises:
        FileNotFoundError: If completion.json, the metadata or the GGUF is missing.
        ValueError: If 0001 did not complete, the load check did not pass, or fields are missing.
    """
    completion = upstream_dir / "completion.json"
    meta_path = upstream_dir / meta_filename
    gguf_path = upstream_dir / gguf_filename
    for path in (completion, meta_path, gguf_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing 0001 output: {path}")
    status = json.loads(completion.read_text()).get("status")
    if status != "completed":
        raise ValueError(f"0001 variant not completed (status={status!r}): {upstream_dir}")
    meta = json.loads(meta_path.read_text())
    if meta.get("status") != "completed" or meta.get("load_check") != "passed":
        raise ValueError(f"0001 metadata not completed/passed: status={meta.get('status')!r} load_check={meta.get('load_check')!r}")
    if meta.get("gguf_file") != gguf_filename:
        raise ValueError(f"0001 metadata gguf_file={meta.get('gguf_file')!r}, expected {gguf_filename!r}")
    for key in ("sha256", "size_bytes", "tensor_count"):
        if key not in meta:
            raise ValueError(f"0001 metadata lacks {key!r}: {meta_path}")
    return meta


def verify_upstream_gguf(gguf_path: Path, upstream_meta: dict[str, Any], chunk_bytes: int) -> str:
    """Recompute size and SHA-256 of the 0001 GGUF and compare with its metadata.

    Returns:
        The verified SHA-256.

    Raises:
        ValueError: On size or SHA-256 mismatch.
    """
    size = gguf_path.stat().st_size
    if size != int(upstream_meta["size_bytes"]):
        raise ValueError(f"Input size mismatch: {size} != recorded {upstream_meta['size_bytes']} ({gguf_path})")
    sha256 = sha256_file(gguf_path, chunk_bytes)
    if sha256 != upstream_meta["sha256"]:
        raise ValueError(f"Input SHA-256 mismatch: {sha256} != recorded {upstream_meta['sha256']} ({gguf_path})")
    return sha256


def build_quantize_command(config: dict, src: Path, tmp_path: Path, qtype: str) -> list[str]:
    """Build the llama-quantize command line (preprocessing.steps[1]); no imatrix."""
    return [config["quantize_bin"], str(src), str(tmp_path), qtype, str(int(config["quantize_threads"]))]


def child_env() -> dict[str, str]:
    """Environment for llama.cpp tools: drop orchestration-venv python vars."""
    return {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> tuple[int, float, float]:
    """Run a command with stdout/stderr streamed to a log file.

    Returns:
        (return code, wall time [s], peak RSS of children so far [MB]).
    """
    t0 = time.monotonic()
    with open(log_path, "w") as log:
        log.write(f"$ {shlex.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    wall = time.monotonic() - t0
    max_rss_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0  # KiB -> MiB
    return proc.returncode, wall, max_rss_mb


def tail_text(path: Path, n_lines: int = 40) -> str:
    """Return the last lines of a text file (for error messages)."""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def parse_quantize_log(text: str) -> dict[str, Any]:
    """Extract the build-info line and reported sizes from llama-quantize output.

    Missing values are returned as None (they are secondary metrics).
    """
    out: dict[str, Any] = {}
    for key, pattern in SIZE_RE.items():
        match = pattern.search(text)
        out[key] = float(match.group(1)) if match else None
    if out["model_size_mib"] and out["quant_size_mib"]:
        out["reported_size_ratio"] = round(out["quant_size_mib"] / out["model_size_mib"], 4)
    else:
        out["reported_size_ratio"] = None
    build_line = next((line.strip() for line in text.splitlines() if "build" in line.lower()), None)
    out["quantize_build_info"] = build_line
    return out


def inspect_gguf(gguf_python: str, gguf_py_path: str, gguf_path: Path, env: dict[str, str]) -> dict[str, Any]:
    """Read GGUF header/tensor info with llama.cpp gguf-py in a subprocess.

    Raises:
        RuntimeError: If the reader exits non-zero (load error / truncated file).
    """
    cmd = [gguf_python, "-I", "-c", GGUF_INSPECT_SCRIPT, gguf_py_path, str(gguf_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"GGUF load check failed for {gguf_path}:\n{proc.stderr[-4000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def validate_inspection(
    info: dict[str, Any], expected_file_type: int, expected_tensor_type: str, expected_tensor_count: int
) -> None:
    """Fail if the quantized GGUF is incomplete or not of the requested type.

    Raises:
        ValueError: On tensor count different from the input, truncated data,
            wrong ``general.file_type`` or no tensor of the expected type.
    """
    if info["tensor_count"] != expected_tensor_count:
        raise ValueError(f"tensor_count={info['tensor_count']} != input tensor_count {expected_tensor_count}")
    if info["data_end"] > info["file_size"]:
        raise ValueError(f"GGUF truncated: tensor data ends at {info['data_end']} > file size {info['file_size']}")
    if info["file_type"] != expected_file_type:
        raise ValueError(f"general.file_type={info['file_type']!r}, expected {expected_file_type}")
    if info["tensor_type_counts"].get(expected_tensor_type, 0) <= 0:
        raise ValueError(f"No {expected_tensor_type} tensors: {info['tensor_type_counts']}")


def write_json(path: Path, data: dict) -> None:
    """Write JSON via a temporary file and rename."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(Path(__file__).parent)
    type_spec = validate_args(config, args.model, args.type)
    variant_key = build_variant_key(args.model, args.type)

    # Exits immediately if already completed (guard checks project_root/outputs).
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    out_name = config["output_filename_template"].format(type=args.type)
    final_path = run_dir / out_name
    tmp_path = run_dir / (out_name + ".tmp")
    if final_path.exists():
        # Left by an earlier attempt that failed after the rename (run_dir is not
        # completed). resume_policy: restart from scratch; the new .tmp replaces it.
        logger.warning(f"Unverified output from an earlier failed attempt will be replaced: {final_path}")

    upstream = config["upstream"]
    upstream_dir = project_root / upstream["bf16_dir"] / args.model
    src_path = upstream_dir / upstream["bf16_filename"]
    chunk = int(config["sha256_chunk_bytes"])

    runtime = read_llamacpp_labels(
        Path(config["sif_labels_file"]), config["expected_llamacpp_ref"], config["expected_llamacpp_commit"]
    )
    upstream_meta = load_upstream_meta(upstream_dir, upstream["bf16_meta_filename"], upstream["bf16_filename"])
    cmd = build_quantize_command(config, src_path, tmp_path, args.type)

    meta: dict[str, Any] = {
        "exp_name": exp_name,
        "variant_key": variant_key,
        "model": args.model,
        "type": args.type,
        "imatrix": None,
        "input_gguf": str(src_path),
        "input_upstream_exp": upstream["bf16_exp"],
        "input_size_bytes": int(upstream_meta["size_bytes"]),
        "input_sha256_recorded": upstream_meta["sha256"],
        "input_tensor_count": int(upstream_meta["tensor_count"]),
        "repo_id": upstream_meta.get("repo_id"),
        "snapshot": upstream_meta.get("snapshot"),
        "quantize_command": shlex.join(cmd),
        "sif_path": os.environ.get("APPTAINER_CONTAINER", ""),
        **runtime,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "status": "verifying_input",
        "started_at": datetime.now().astimezone().isoformat(),
    }
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, model=args.model, type=args.type)
    meta_path = run_dir / "gguf_meta.json"
    write_json(meta_path, meta)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir: {run_dir}")
    logger.info(f"input:   {src_path}")
    logger.info(f"runtime: llama.cpp {runtime['llamacpp_ref']} {runtime['llamacpp_commit']}")

    t0 = time.monotonic()
    input_sha = verify_upstream_gguf(src_path, upstream_meta, chunk)
    meta.update({"input_sha256_verified": input_sha, "input_verify_wall_time_s": round(time.monotonic() - t0, 1), "status": "quantizing"})
    write_json(meta_path, meta)
    logger.info(f"input SHA-256 verified: {input_sha}")
    logger.info(f"command: {meta['quantize_command']}")

    env = child_env()
    log_path = run_dir / "quantize.log"
    returncode, wall, max_rss_mb = run_logged(cmd, log_path, env)
    meta.update({"quantize_returncode": returncode, "quantize_wall_time_s": round(wall, 1), "quantize_max_rss_mb": round(max_rss_mb, 1)})
    meta.update(parse_quantize_log(log_path.read_text(errors="replace")))
    if returncode != 0:
        meta["status"] = "failed"
        write_json(meta_path, meta)
        raise RuntimeError(f"llama-quantize exited {returncode}:\n{tail_text(log_path)}")
    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        meta["status"] = "failed"
        write_json(meta_path, meta)
        raise RuntimeError(f"llama-quantize exited 0 but {tmp_path} is missing or empty")
    os.replace(tmp_path, final_path)
    logger.info(f"Quantized in {wall:.1f} s (child peak RSS {max_rss_mb:.0f} MB) -> {final_path}")

    size_bytes = final_path.stat().st_size
    sha256 = sha256_file(final_path, chunk)
    meta.update({"gguf_file": final_path.name, "size_bytes": size_bytes, "sha256": sha256, "status": "load_check"})
    write_json(meta_path, meta)
    logger.info(f"sha256: {sha256}  size: {size_bytes} bytes")

    info = inspect_gguf(config["gguf_python"], config["gguf_py_path"], final_path, env)
    validate_inspection(info, int(type_spec["file_type"]), type_spec["tensor_type"], int(upstream_meta["tensor_count"]))
    if info["file_size"] != size_bytes:
        raise ValueError(f"File size changed during checks: {info['file_size']} != {size_bytes}")
    meta.update(
        {
            "tensor_count": info["tensor_count"],
            "tensor_type_counts": info["tensor_type_counts"],
            "file_type": info["file_type"],
            "architecture": info["architecture"],
            "kv_count": info["kv_count"],
            "load_check": "passed",
            "load_check_reader": info["gguf_module"],
            "status": "completed",
            "finished_at": datetime.now().astimezone().isoformat(),
        }
    )
    write_json(meta_path, meta)
    logger.info(f"load check: {info['tensor_count']} tensors, types {info['tensor_type_counts']}, file_type {info['file_type']}")

    row = {
        "variant_key": variant_key,
        "model": args.model,
        "type": args.type,
        "input_sha256": input_sha,
        "gguf_file": final_path.name,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "tensor_count": info["tensor_count"],
        "tensor_type_counts": json.dumps(info["tensor_type_counts"]),
        "file_type": info["file_type"],
        "quantize_wall_time_s": meta["quantize_wall_time_s"],
        "quantize_max_rss_mb": meta["quantize_max_rss_mb"],
        "model_size_mib": meta["model_size_mib"],
        "quant_size_mib": meta["quant_size_mib"],
        "reported_size_ratio": meta["reported_size_ratio"],
        "llamacpp_commit": runtime["llamacpp_commit"],
        "load_check": "passed",
    }
    pd.DataFrame([row]).to_parquet(run_dir / "results.parquet", index=False)

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
