"""0001 preprocess_gguf_bf16: convert one fixed HF snapshot to a BF16 GGUF.

One job handles one model (variant_key == plan task id, e.g. ``qwen3-8b``):

1. Verify the runtime (llama.cpp ref/commit from the SIF labels).
2. Verify the fixed HF snapshot and that every weight shard is present.
3. Run llama.cpp ``convert_hf_to_gguf.py --outtype bf16`` into ``model-BF16.gguf.tmp``.
4. Rename ``.tmp`` to ``model-BF16.gguf`` only after exit 0.
5. Record SHA-256 / size, then run a metadata load check with llama.cpp gguf-py
   (``GGUFReader``: tensor count, tensor types, ``general.file_type``).
6. Write ``gguf_meta.json`` (read by 0002) and ``results.parquet``.
"""

import argparse
import hashlib
import json
import logging
import os
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
    """Parse the swept dimension (must match GRID_ARGS in run_slurm.sh).

    Args:
        argv: Argument list (None uses sys.argv).

    Returns:
        Parsed arguments with ``model``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model id from config.yml models (variant id).")
    return parser.parse_args(argv)


def build_variant_key(model: str) -> str:
    """Return the variant key; equals the plan task id (e.g. ``qwen3-8b``)."""
    return model


def get_model_spec(config: dict, model: str) -> dict:
    """Look up the fixed repo id / snapshot for a model id.

    Raises:
        KeyError: If the model is not defined in config.yml.
        ValueError: If a required field is missing.
    """
    models = config["models"]
    if model not in models:
        raise KeyError(f"Unknown --model {model!r}; allowed: {sorted(models)}")
    spec = models[model]
    for key in ("repo_id", "snapshot"):
        if not spec.get(key):
            raise ValueError(f"config.yml models.{model}.{key} is empty")
    return spec


def resolve_snapshot_dir(hf_hub_dir: Path, repo_id: str, snapshot: str) -> Path:
    """Return the HF cache snapshot directory for a fixed revision.

    Raises:
        FileNotFoundError: If the snapshot directory does not exist.
    """
    snap_dir = hf_hub_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots" / snapshot
    if not snap_dir.is_dir():
        raise FileNotFoundError(f"Snapshot directory not found: {snap_dir}")
    return snap_dir


def verify_shards(snapshot_dir: Path) -> list[dict[str, Any]]:
    """Check that every safetensors shard referenced by the snapshot exists and is non-empty.

    Uses ``model.safetensors.index.json`` when present, otherwise ``model.safetensors``.

    Args:
        snapshot_dir: HF snapshot directory.

    Returns:
        One dict per shard with ``name`` and ``size_bytes``.

    Raises:
        FileNotFoundError: If the index or a shard is missing.
        ValueError: If a shard is empty or the index lists no shards.
    """
    index_path = snapshot_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        names = sorted(set(weight_map.values()))
    elif (snapshot_dir / "model.safetensors").exists():
        names = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No safetensors index or model.safetensors in {snapshot_dir}")
    if not names:
        raise ValueError(f"Index lists no shards: {index_path}")
    shards: list[dict[str, Any]] = []
    for name in names:
        shard = snapshot_dir / name
        if not shard.exists():  # follows the HF cache symlink to the blob
            raise FileNotFoundError(f"Missing shard: {shard}")
        size = shard.stat().st_size
        if size == 0:
            raise ValueError(f"Empty shard: {shard}")
        shards.append({"name": name, "size_bytes": size})
    return shards


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


def build_convert_command(config: dict, snapshot_dir: Path, tmp_path: Path) -> list[str]:
    """Build the converter command line (preprocessing.steps[0])."""
    return [
        config["converter_python"],
        config["converter_script"],
        str(snapshot_dir),
        "--outtype",
        config["outtype"],
        "--outfile",
        str(tmp_path),
    ]


def child_env() -> dict[str, str]:
    """Environment for llama.cpp tools: drop orchestration-venv python vars, force offline HF."""
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> tuple[int, float, float]:
    """Run a command with stdout/stderr streamed to a log file.

    Args:
        cmd: Command argv.
        log_path: File that receives the combined output (written as it runs).
        env: Environment for the child.

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


def validate_inspection(info: dict[str, Any], expected_file_type: int, expected_tensor_type: str) -> None:
    """Fail if the GGUF metadata is not a complete file of the expected type.

    Raises:
        ValueError: On zero tensors, truncated data, wrong ``general.file_type`` or
            no tensor of the expected type.
    """
    if info["tensor_count"] <= 0:
        raise ValueError("GGUF has no tensors")
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
    spec = get_model_spec(config, args.model)
    variant_key = build_variant_key(args.model)

    # Exits immediately if already completed (guard checks project_root/outputs).
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    final_path = run_dir / config["output_filename"]
    tmp_path = run_dir / (config["output_filename"] + ".tmp")
    if final_path.exists():
        # Left by an earlier attempt that failed after the rename (run_dir is not
        # completed). resume_policy: restart from scratch; the new .tmp replaces it.
        logger.warning(f"Unverified output from an earlier failed attempt will be replaced: {final_path}")

    runtime = read_llamacpp_labels(
        Path(config["sif_labels_file"]), config["expected_llamacpp_ref"], config["expected_llamacpp_commit"]
    )
    snapshot_dir = resolve_snapshot_dir(Path(config["hf_hub_dir"]), spec["repo_id"], spec["snapshot"])
    shards = verify_shards(snapshot_dir)
    cmd = build_convert_command(config, snapshot_dir, tmp_path)

    meta: dict[str, Any] = {
        "exp_name": exp_name,
        "variant_key": variant_key,
        "model": args.model,
        "repo_id": spec["repo_id"],
        "snapshot": spec["snapshot"],
        "snapshot_dir": str(snapshot_dir),
        "shards": shards,
        "outtype": config["outtype"],
        "converter_command": shlex.join(cmd),
        "sif_path": os.environ.get("APPTAINER_CONTAINER", ""),
        **runtime,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "status": "converting",
        "started_at": datetime.now().astimezone().isoformat(),
    }
    write_run_metadata(run_dir, **{k: meta[k] for k in ("exp_name", "variant_key", "model", "repo_id", "snapshot", "outtype")})
    meta_path = run_dir / "gguf_meta.json"
    write_json(meta_path, meta)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:  {run_dir}")
    logger.info(f"snapshot: {snapshot_dir} ({len(shards)} shards)")
    logger.info(f"runtime:  llama.cpp {runtime['llamacpp_ref']} {runtime['llamacpp_commit']}")
    logger.info(f"command:  {meta['converter_command']}")

    env = child_env()
    returncode, wall, max_rss_mb = run_logged(cmd, run_dir / "convert.log", env)
    meta.update({"convert_returncode": returncode, "convert_wall_time_s": round(wall, 1), "convert_max_rss_mb": round(max_rss_mb, 1)})
    if returncode != 0:
        meta["status"] = "failed"
        write_json(meta_path, meta)
        raise RuntimeError(f"Converter exited {returncode}:\n{tail_text(run_dir / 'convert.log')}")
    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        meta["status"] = "failed"
        write_json(meta_path, meta)
        raise RuntimeError(f"Converter exited 0 but {tmp_path} is missing or empty")
    os.replace(tmp_path, final_path)
    logger.info(f"Converted in {wall:.1f} s (child peak RSS {max_rss_mb:.0f} MB) -> {final_path}")

    size_bytes = final_path.stat().st_size
    sha256 = sha256_file(final_path, int(config["sha256_chunk_bytes"]))
    meta.update({"gguf_file": final_path.name, "size_bytes": size_bytes, "sha256": sha256, "status": "load_check"})
    write_json(meta_path, meta)
    logger.info(f"sha256: {sha256}  size: {size_bytes} bytes")

    info = inspect_gguf(config["converter_python"], config["gguf_py_path"], final_path, env)
    validate_inspection(info, int(config["expected_file_type"]), config["expected_tensor_type"])
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
        "repo_id": spec["repo_id"],
        "snapshot": spec["snapshot"],
        "outtype": config["outtype"],
        "gguf_file": final_path.name,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "tensor_count": info["tensor_count"],
        "tensor_type_counts": json.dumps(info["tensor_type_counts"]),
        "file_type": info["file_type"],
        "convert_wall_time_s": meta["convert_wall_time_s"],
        "convert_max_rss_mb": meta["convert_max_rss_mb"],
        "llamacpp_commit": runtime["llamacpp_commit"],
        "load_check": "passed",
    }
    pd.DataFrame([row]).to_parquet(run_dir / "results.parquet", index=False)

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
