"""0004 extract_hidden_states: save per-token hidden states of Qwen3-8B from HF, vLLM and llama.cpp.

One job processes exactly one task (``--task``), see plan.md ``variants.tasks``:

=================== ========================================================================
task                method
=================== ========================================================================
hf-bf16             HF transformers, bfloat16, ``output_hidden_states=True`` + hook (baseline)
hf-fp32             same in float32 (precision floor)
vllm-bf16-hidden    vLLM ``extract_hidden_states`` + ``ExampleHiddenStatesConnector``
vllm-bf16-pooling   vLLM ``runner="pooling"``, task ``token_embed``, ``use_activation=False``
llamacpp-bf16       llama.cpp ``dump_hidden`` (``cb_eval``) on the 0001 BF16 GGUF
llamacpp-q8_0       same on the 0002 Q8_0 GGUF
llamacpp-q4_k_m     same on the 0002 Q4_K_M GGUF
=================== ========================================================================

Layer naming (shared by all engines; ``N`` = num_hidden_layers = 36):

- ``emb``            embedding output = input of block 1.
                     HF ``hidden_states[0]``; vLLM aux layer id 0; llama.cpp ``inp_embd``.
- ``pre_L01..pre_L36`` residual stream after block k (k = 1..N), BEFORE the final RMSNorm.
                     HF ``hidden_states[k]`` for k = 1..N-1 and a forward hook on
                     ``model.layers[N-1]`` for k = N (transformers 5.x replaces
                     ``hidden_states[N]`` with the normed ``last_hidden_state``);
                     vLLM aux layer id k (``hidden + residual`` after block k);
                     llama.cpp ``l_out-(k-1)``.
- ``final_norm``     output of the final RMSNorm. HF ``last_hidden_state`` (= ``hidden_states[N]``);
                     vLLM pooling output; llama.cpp ``result_norm`` via the embeddings API.
- ``final_norm_cb``  llama.cpp only: ``result_norm`` captured by ``cb_eval`` (self-consistency).
- ``last_logits``    logits of the last prompt token, shape (vocab,). HF ``logits[:, -1]``;
                     vLLM raw logits of the first sampled position (``logprobs_mode=raw_logits``);
                     llama.cpp ``llama_get_logits_ith(T-1)``.

Outputs (plan.outputs): ``<run_dir>/<prompt_id>/<layer>.npy`` (float32),
``<run_dir>/<prompt_id>/token_ids.json``, ``<run_dir>/capability.parquet`` and
``<run_dir>/run_info.json``. A prompt is finished when ``prompt_meta.json`` with
``status == "ok"`` exists (written last); resumed runs skip finished prompts.

Failure policy: problems with inputs, environment or the build fail fast (non-zero exit).
An engine method that raises, returns wrong shapes/token IDs or non-finite values is
recorded as unavailable in capability.parquet (plan.hypothesis.decision_rules) and the
method stops (plan.stop_conditions.correctness_failure).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

TASK_IDS: tuple[str, ...] = (
    "hf-bf16",
    "hf-fp32",
    "vllm-bf16-hidden",
    "vllm-bf16-pooling",
    "llamacpp-bf16",
    "llamacpp-q8_0",
    "llamacpp-q4_k_m",
)
EMB = "emb"
FINAL_NORM = "final_norm"
FINAL_NORM_CB = "final_norm_cb"
LAST_LOGITS = "last_logits"
PROMPT_META = "prompt_meta.json"
TOKEN_IDS = "token_ids.json"


class MethodFailure(Exception):
    """An engine method could not deliver the specified arrays (recorded as unavailable)."""


# ── Small helpers ─────────────────────────────────────────────────────────────


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
        raise FileNotFoundError(f"missing {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path} is not a mapping")
    return cfg


def parse_args() -> argparse.Namespace:
    """Define CLI args for all variable dimensions (used in GRID_ARGS)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASK_IDS)
    return parser.parse_args()


def pre_layer(k: int) -> str:
    """Name of the pre-norm residual stream after block k (1-indexed)."""
    return f"pre_L{k:02d}"


def expected_layers(task_cfg: dict[str, Any], n_layers: int) -> list[str]:
    """Return the layer names a task is specified to deliver (plan.variants.tasks.layers)."""
    blocks = [EMB] + [pre_layer(k) for k in range(1, n_layers + 1)]
    engine = task_cfg["engine"]
    if engine == "hf":
        return blocks + [FINAL_NORM, LAST_LOGITS]
    if engine == "vllm" and task_cfg["mode"] == "extract_hidden_states":
        return blocks + [LAST_LOGITS]
    if engine == "vllm" and task_cfg["mode"] == "pooling":
        return [FINAL_NORM]
    if engine == "llamacpp":
        return blocks + [FINAL_NORM, FINAL_NORM_CB, LAST_LOGITS]
    raise ValueError(f"unknown task config {task_cfg}")


def norm_state(layer: str) -> str:
    """Normalization state of a layer name (used to align pre-norm vs normed in 0005)."""
    if layer == EMB:
        return "embedding"
    if layer.startswith("pre_L"):
        return "pre_norm"
    if layer in (FINAL_NORM, FINAL_NORM_CB):
        return "normed"
    if layer == LAST_LOGITS:
        return "logits"
    raise ValueError(f"unknown layer {layer}")


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    """SHA-256 hex digest of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically (tmp file + rename)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """Save a float32 array as .npy atomically."""
    if arr.dtype != np.float32:
        raise TypeError(f"{path.name}: expected float32, got {arr.dtype}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.save(f, np.ascontiguousarray(arr), allow_pickle=False)
    os.replace(tmp, path)


def git_revision(project_root: Path) -> str:
    """Return the repository HEAD commit, or 'unknown' if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def check_array(arr: np.ndarray, layer: str, n_tokens: int, hidden: int) -> tuple[bool, str, str | None, bool]:
    """Validate one extracted array against the plan's quality checks.

    Args:
        arr: Extracted array.
        layer: Layer name (``last_logits`` is 1-D, all others are (n_tokens, hidden)).
        n_tokens: Number of prompt tokens.
        hidden: Hidden size.

    Returns:
        (available, shape string, error or None, all_finite).
    """
    shape = "x".join(str(s) for s in arr.shape)
    finite = bool(np.isfinite(arr).all())
    if layer == LAST_LOGITS:
        ok_shape = arr.ndim == 1 and arr.shape[0] > 0
        expect = "(vocab,)"
    else:
        ok_shape = arr.shape == (n_tokens, hidden)
        expect = f"({n_tokens}, {hidden})"
    if not ok_shape:
        return False, shape, f"shape {arr.shape} != expected {expect}", finite
    if not finite:
        return False, shape, "non-finite values", finite
    return True, shape, None, finite


# ── Prompt bookkeeping ────────────────────────────────────────────────────────


class PromptSet:
    """Prompts, their HF token IDs and per-prompt output bookkeeping for one task."""

    def __init__(
        self,
        prompts: list[dict[str, Any]],
        token_ids: dict[str, list[int]],
        run_dir: Path,
        task: str,
        task_cfg: dict[str, Any],
        n_layers: int,
        hidden: int,
    ) -> None:
        self.prompts = prompts
        self.token_ids = token_ids
        self.run_dir = run_dir
        self.task = task
        self.task_cfg = task_cfg
        self.n_layers = n_layers
        self.hidden = hidden
        self.layers = expected_layers(task_cfg, n_layers)

    def prompt_dir(self, pid: str) -> Path:
        """Output directory of one prompt."""
        return self.run_dir / pid

    def is_done(self, pid: str) -> bool:
        """True when the prompt was finished successfully in this or an earlier attempt."""
        meta = self.prompt_dir(pid) / PROMPT_META
        if not meta.exists():
            return False
        return json.loads(meta.read_text())["status"] == "ok"

    def todo(self) -> list[dict[str, Any]]:
        """Prompts that still need extraction."""
        return [p for p in self.prompts if not self.is_done(p["id"])]

    def write_ok(
        self,
        pid: str,
        arrays: dict[str, np.ndarray] | None,
        engine_token_ids: list[int],
        seconds: dict[str, float],
        timing_scope: str,
        sources: dict[str, str],
        extra_token_info: dict[str, Any] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> bool:
        """Validate and save one prompt; returns False if any layer failed validation.

        Args:
            pid: Prompt id.
            arrays: Layer name -> array to save (None when the files were already written,
                e.g. by dump_hidden; they are then loaded for validation).
            engine_token_ids: Token IDs the engine reports it processed.
            seconds: Layer name -> engine seconds of the pass producing that layer.
            timing_scope: ``prompt`` (per-prompt pass) or ``batch`` (one call for all prompts).
            sources: Layer name -> engine-specific source description.
            extra_token_info: Additional fields for token_ids.json.
            extra_meta: Additional fields for prompt_meta.json.
        """
        d = self.prompt_dir(pid)
        d.mkdir(parents=True, exist_ok=True)
        hf_ids = self.token_ids[pid]
        ids_equal = list(engine_token_ids) == list(hf_ids)
        rows: list[dict[str, Any]] = []
        all_ok = True
        for layer in self.layers:
            path = d / f"{layer}.npy"
            if arrays is not None:
                if layer not in arrays:
                    raise KeyError(f"{pid}: layer {layer} missing from engine output")
                arr = np.asarray(arrays[layer], dtype=np.float32)
                atomic_save_npy(path, arr)
            else:
                if not path.exists():
                    rows.append(self._row(pid, layer, False, "", seconds.get(layer), timing_scope,
                                          f"{path.name} not written", None, ids_equal, sources.get(layer)))
                    all_ok = False
                    continue
                arr = np.load(path, mmap_mode="r", allow_pickle=False)
                if arr.dtype != np.float32:
                    raise TypeError(f"{path}: dtype {arr.dtype}, expected float32")
            available, shape, err, finite = check_array(arr, layer, len(hf_ids), self.hidden)
            if not ids_equal:
                available, err = False, "engine token IDs differ from HF token IDs"
            all_ok = all_ok and available
            rows.append(self._row(pid, layer, available, shape, seconds.get(layer), timing_scope,
                                  err, finite, ids_equal, sources.get(layer)))
        token_info = {"prompt_id": pid, "token_ids": list(engine_token_ids), "hf_token_ids": list(hf_ids),
                      "token_ids_equal_hf": ids_equal}
        token_info.update(extra_token_info or {})
        atomic_write_text(d / TOKEN_IDS, json.dumps(token_info))
        meta = {"prompt_id": pid, "status": "ok" if all_ok else "error", "task": self.task,
                "n_tokens": len(hf_ids), "rows": rows}
        meta.update(extra_meta or {})
        atomic_write_text(d / PROMPT_META, json.dumps(meta, indent=1))
        return all_ok

    def write_error(self, pid: str, error: str) -> None:
        """Record that a prompt could not be extracted (no arrays)."""
        d = self.prompt_dir(pid)
        d.mkdir(parents=True, exist_ok=True)
        rows = [self._row(pid, layer, False, "", None, None, error, None, None, None) for layer in self.layers]
        meta = {"prompt_id": pid, "status": "error", "task": self.task,
                "n_tokens": len(self.token_ids[pid]), "rows": rows}
        atomic_write_text(d / PROMPT_META, json.dumps(meta, indent=1))

    def _row(
        self, pid: str, layer: str, available: bool, shape: str, seconds: float | None,
        timing_scope: str | None, error: str | None, finite: bool | None, ids_equal: bool | None,
        source: str | None,
    ) -> dict[str, Any]:
        return {
            "variant": self.task,
            "method": self.task_cfg.get("mode", self.task_cfg["engine"]),
            "engine": self.task_cfg["engine"],
            "prompt_id": pid,
            "layer": layer,
            "norm_state": norm_state(layer),
            "available": bool(available),
            "shape": shape,
            "n_tokens": len(self.token_ids[pid]),
            "seconds": float(seconds) if seconds is not None else float("nan"),
            "timing_scope": timing_scope,
            "error": error,
            "all_finite": finite,
            "token_ids_equal_hf": ids_equal,
            "source": source,
        }

    def write_capability(self) -> Path:
        """Collect all prompt_meta.json rows into capability.parquet."""
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for p in self.prompts:
            meta_path = self.prompt_dir(p["id"]) / PROMPT_META
            if not meta_path.exists():
                raise RuntimeError(f"prompt {p['id']} has no record; every prompt needs arrays or an error")
            rows.extend(json.loads(meta_path.read_text())["rows"])
        df = pd.DataFrame(rows)
        out = self.run_dir / "capability.parquet"
        tmp = out.with_name(f".{out.name}.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        return out


# ── HF transformers ───────────────────────────────────────────────────────────


def run_hf(ps: PromptSet, cfg: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    """Extract all layers with HF transformers (fail fast on any error: this is the reference)."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    task_cfg = ps.task_cfg
    torch.manual_seed(int(cfg["seed"]))
    torch.set_float32_matmul_precision("highest")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[task_cfg["dtype"]]
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_snapshot"], dtype=dtype, device_map="cuda",
        attn_implementation=task_cfg["attn_implementation"], local_files_only=True,
    )
    model.eval()
    load_s = time.perf_counter() - t0
    n_layers = ps.n_layers
    if model.config.num_hidden_layers != n_layers or model.config.hidden_size != ps.hidden:
        raise RuntimeError(f"model config {model.config.num_hidden_layers}/{model.config.hidden_size} "
                           f"differs from config.yml {n_layers}/{ps.hidden}")
    if model.dtype != dtype:
        raise RuntimeError(f"model dtype {model.dtype} != requested {dtype}")

    captured: dict[int, Any] = {}

    def make_hook(idx: int):
        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            t = out[0] if isinstance(out, tuple) else out
            captured[idx] = t.detach()
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(model.model.layers)]

    def forward(ids: list[int]) -> tuple[Any, float]:
        x = torch.tensor([ids], device="cuda")
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=x, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        return out, time.perf_counter() - t

    # Warm-up (not timed, not saved).
    captured.clear()
    forward(ps.token_ids[ps.prompts[0]["id"]])

    sources = {EMB: "hidden_states[0]", FINAL_NORM: f"hidden_states[{n_layers}] (= last_hidden_state, normed)",
               LAST_LOGITS: "logits[0, -1] (logits_to_keep=1)"}
    for k in range(1, n_layers):
        sources[pre_layer(k)] = f"hidden_states[{k}]"
    sources[pre_layer(n_layers)] = f"forward hook on model.layers[{n_layers - 1}] output"

    for p in tqdm(ps.todo(), desc=f"{ps.task} prompts"):
        pid = p["id"]
        ids = ps.token_ids[pid]
        captured.clear()
        out, fwd_s = forward(ids)
        hs = out.hidden_states
        if len(hs) != n_layers + 1:
            raise RuntimeError(f"{pid}: hidden_states has {len(hs)} entries, expected {n_layers + 1}")
        if sorted(captured) != list(range(n_layers)):
            raise RuntimeError(f"{pid}: hooks captured {sorted(captured)}")
        # Layer-mapping assumption: hidden_states[k] == output of block k for k = 1..N-1.
        for k in range(1, n_layers):
            if not torch.equal(hs[k], captured[k - 1]):
                raise RuntimeError(f"{pid}: hidden_states[{k}] differs from block {k} output; mapping invalid")
        with torch.inference_mode():
            last_is_norm = bool(torch.equal(hs[n_layers], model.model.norm(captured[n_layers - 1])))
            last_equals_pre = bool(torch.equal(hs[n_layers], captured[n_layers - 1]))
        arrays = {EMB: hs[0][0].float().cpu().numpy()}
        for k in range(1, n_layers + 1):
            arrays[pre_layer(k)] = captured[k - 1][0].float().cpu().numpy()
        arrays[FINAL_NORM] = hs[n_layers][0].float().cpu().numpy()
        arrays[LAST_LOGITS] = out.logits[0, -1].float().cpu().numpy()
        ok = ps.write_ok(
            pid, arrays, ids, {layer: fwd_s for layer in ps.layers}, "prompt", sources,
            extra_meta={"hidden_states_last_is_norm_of_pre_L36": last_is_norm,
                        "hidden_states_last_equals_pre_L36": last_equals_pre,
                        "forward_s": fwd_s},
        )
        if not ok:
            raise RuntimeError(f"{pid}: reference arrays failed validation; see {ps.prompt_dir(pid) / PROMPT_META}")
        del out, hs, arrays
        captured.clear()

    for h in handles:
        h.remove()
    return {
        "load_s": load_s,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "max_gpu_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "model_dtype": str(model.dtype),
        "attn_implementation": task_cfg["attn_implementation"],
    }


# ── vLLM ──────────────────────────────────────────────────────────────────────


def run_vllm(ps: PromptSet, cfg: dict[str, Any], scratch: Path, logger: logging.Logger) -> dict[str, Any]:
    """Extract with vLLM (``extract_hidden_states`` or pooling). Engine errors -> MethodFailure."""
    import torch
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt

    mode = ps.task_cfg["mode"]
    vcfg = cfg["vllm"]
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    n_layers = ps.n_layers
    todo = ps.todo()
    info: dict[str, Any] = {"vllm": vllm.__version__, "torch": torch.__version__, "mode": mode}
    if not todo:
        return info
    common: dict[str, Any] = dict(
        model=cfg["hf_snapshot"], tokenizer=cfg["hf_snapshot"], dtype=ps.task_cfg["dtype"],
        max_model_len=int(vcfg["max_model_len"]),
        enable_chunked_prefill=bool(vcfg["enable_chunked_prefill"]),
        enable_prefix_caching=bool(vcfg["enable_prefix_caching"]),
        max_num_batched_tokens=int(vcfg["max_num_batched_tokens"]),
        gpu_memory_utilization=float(vcfg["gpu_memory_utilization"]),
        seed=seed,
    )
    t0 = time.perf_counter()
    hs_dir: Path | None = None
    try:
        if mode == "extract_hidden_states":
            from vllm.config.kv_transfer import KVTransferConfig

            ecfg = vcfg["extract_hidden_states"]
            hs_dir = scratch / "vllm_hidden_states"
            hs_dir.mkdir(parents=True, exist_ok=True)
            layer_ids = list(range(0, n_layers + 1))  # 0 = emb, k = after block k (pre-norm)
            llm = LLM(
                **common,
                max_logprobs=int(ecfg["max_logprobs"]),
                logprobs_mode=ecfg["logprobs_mode"],
                speculative_config={
                    "method": "extract_hidden_states",
                    "num_speculative_tokens": 1,
                    "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": layer_ids}},
                },
                kv_transfer_config=KVTransferConfig(
                    kv_connector="ExampleHiddenStatesConnector",
                    kv_role="kv_producer",
                    kv_connector_extra_config={"shared_storage_path": str(hs_dir)},
                ),
            )
            info["layer_ids"] = layer_ids
        elif mode == "pooling":
            from vllm.config import PoolerConfig

            llm = LLM(**common, runner="pooling", pooler_config=PoolerConfig(task=vcfg["pooling"]["task"]))
        else:
            raise ValueError(f"unknown vLLM mode {mode}")
    except Exception as e:  # engine construction is part of the method under test
        raise MethodFailure(f"vLLM engine construction failed: {e!r}") from e
    info["load_s"] = time.perf_counter() - t0
    info["engine_args"] = {k: v for k, v in common.items() if k not in ("model", "tokenizer")}

    reqs = [TokensPrompt(prompt_token_ids=ps.token_ids[p["id"]]) for p in todo]

    def run_once() -> list[Any]:
        if mode == "extract_hidden_states":
            ecfg = vcfg["extract_hidden_states"]
            sp = SamplingParams(max_tokens=int(ecfg["max_tokens"]), temperature=float(ecfg["temperature"]),
                                logprobs=int(ecfg["logprobs"]))
            return llm.generate(reqs, sp, use_tqdm=False)
        from vllm.pooling_params import PoolingParams

        pp = PoolingParams(use_activation=bool(vcfg["pooling"]["use_activation"]))
        return llm.encode(reqs, pp, pooling_task=vcfg["pooling"]["task"], use_tqdm=False)

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector as ehc

        warm = run_once()  # warm-up (compilation / CUDA graphs), not timed, not saved
        if mode == "extract_hidden_states":
            for o in warm:
                path = o.kv_transfer_params["hidden_states_path"]
                ehc.load_hidden_states(path)  # wait for the async write before removing it
                ehc.cleanup_hidden_states(path)
        del warm
        torch.cuda.synchronize()
        t = time.perf_counter()
        outs = run_once()
        engine_s = time.perf_counter() - t
    except Exception as e:
        raise MethodFailure(f"vLLM run failed: {e!r}") from e
    info["engine_s_all_prompts"] = engine_s
    info["n_prompts_in_call"] = len(todo)
    info["n_tokens_in_call"] = sum(len(ps.token_ids[p["id"]]) for p in todo)
    if len(outs) != len(todo):
        raise MethodFailure(f"vLLM returned {len(outs)} outputs for {len(todo)} prompts")

    if mode == "extract_hidden_states":
        sources = {EMB: "aux layer id 0 (embedding)", LAST_LOGITS: "raw logits of first sampled position"}
        for k in range(1, n_layers + 1):
            sources[pre_layer(k)] = f"aux layer id {k} (hidden + residual after block {k})"
    else:
        sources = {FINAL_NORM: "pooling token_embed, use_activation=False (after final RMSNorm)"}
    seconds = {layer: engine_s for layer in ps.layers}

    for p, o in tqdm(list(zip(todo, outs)), desc=f"{ps.task} save"):
        pid = p["id"]
        ids = ps.token_ids[pid]
        engine_ids = list(o.prompt_token_ids)
        if engine_ids != ids:
            raise MethodFailure(f"{pid}: vLLM prompt_token_ids differ from HF token IDs")
        if mode == "extract_hidden_states":
            path = o.kv_transfer_params["hidden_states_path"]
            obj = ehc.load_hidden_states(path)
            saved_ids = obj["token_ids"].tolist()
            hs = obj["hidden_states"]  # [T, n_layer_ids, H]
            if saved_ids != ids:
                raise MethodFailure(f"{pid}: connector token_ids differ from HF token IDs")
            if tuple(hs.shape) != (len(ids), n_layers + 1, ps.hidden):
                raise MethodFailure(f"{pid}: connector hidden_states shape {tuple(hs.shape)}, expected "
                                    f"({len(ids)}, {n_layers + 1}, {ps.hidden})")
            hs32 = hs.float().numpy()
            arrays = {EMB: hs32[:, 0, :]}
            for k in range(1, n_layers + 1):
                arrays[pre_layer(k)] = hs32[:, k, :]
            lp = o.outputs[0].logprobs[0]
            vocab = len(lp)
            logits = np.full(vocab, np.nan, dtype=np.float32)
            for tok, v in lp.items():
                if not 0 <= tok < vocab:
                    raise MethodFailure(f"{pid}: logprob token id {tok} outside [0, {vocab})")
                logits[tok] = v.logprob
            arrays[LAST_LOGITS] = logits
            extra = {"vllm_first_sampled_token": int(o.outputs[0].token_ids[0]), "logits_vocab": vocab}
        else:
            data = o.outputs.data
            arrays = {FINAL_NORM: data.float().cpu().numpy()}
            extra = {}
        ok = ps.write_ok(pid, arrays, engine_ids, seconds, "batch", sources, extra_meta=extra)
        if mode == "extract_hidden_states":
            ehc.cleanup_hidden_states(path)
        if not ok:
            raise MethodFailure(f"{pid}: arrays failed validation (shape, token IDs or non-finite)")
    del llm
    return info


# ── llama.cpp ─────────────────────────────────────────────────────────────────


def verify_gguf(project_root: Path, rel: str, logger: logging.Logger) -> dict[str, Any]:
    """Check that the GGUF exists and matches the sha256 in its gguf_meta.json (fail fast)."""
    gguf = project_root / rel
    meta_path = gguf.parent / "gguf_meta.json"
    if not gguf.is_file():
        raise FileNotFoundError(f"GGUF not found: {gguf}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"gguf_meta.json not found next to {gguf}")
    meta = json.loads(meta_path.read_text())
    expected = meta.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{meta_path}: missing or invalid 'sha256'")
    upstream_completion = gguf.parent / "completion.json"
    if not upstream_completion.is_file() or json.loads(upstream_completion.read_text()).get("status") != "completed":
        raise RuntimeError(f"upstream run {gguf.parent} is not completed")
    t = time.perf_counter()
    actual = sha256_file(gguf)
    logger.info(f"GGUF sha256 {actual} ({time.perf_counter() - t:.1f}s)")
    if actual != expected:
        raise ValueError(f"GGUF sha256 mismatch for {gguf}: {actual} != {expected}")
    return {"gguf": str(gguf), "gguf_sha256": actual, "gguf_meta": meta}


def build_dump_hidden(project_root: Path, cfg: dict[str, Any], scratch: Path, logger: logging.Logger) -> Path:
    """Build dump_hidden against /opt/llama.cpp into this job's scratch dir (fail fast)."""
    lcfg = cfg["llamacpp"]
    src = project_root / lcfg["source_dir"]
    build = scratch / "dump_hidden_build"
    env = dict(os.environ)
    env["LIBRARY_PATH"] = f"{lcfg['cuda_stubs']}:{env.get('LIBRARY_PATH', '')}".rstrip(":")
    log = scratch / "dump_hidden_build.log"
    with open(log, "w") as f:
        for cmd in (
            ["cmake", "-S", str(src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
            ["cmake", "--build", str(build), "-j", str(int(lcfg["build_jobs"]))],
        ):
            f.write(f"$ {' '.join(cmd)}\n")
            f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
            if rc != 0:
                tail = log.read_text()[-4000:]
                raise RuntimeError(f"dump_hidden build failed (rc={rc}):\n{tail}")
    binary = build / "dump_hidden"
    if not binary.is_file():
        raise FileNotFoundError(f"build produced no {binary}")
    return binary


def run_llamacpp(
    ps: PromptSet, cfg: dict[str, Any], project_root: Path, scratch: Path, logger: logging.Logger,
) -> dict[str, Any]:
    """Extract with the dump_hidden tool. Preconditions fail fast; engine errors -> MethodFailure."""
    lcfg = cfg["llamacpp"]
    info: dict[str, Any] = verify_gguf(project_root, ps.task_cfg["gguf"], logger)
    binary = build_dump_hidden(project_root, cfg, scratch, logger)
    info["dump_hidden_sha256"] = sha256_file(binary)
    info["dump_hidden_source_sha256"] = {
        name: sha256_file(project_root / lcfg["source_dir"] / name) for name in ("dump_hidden.cpp", "CMakeLists.txt")
    }
    logger.info(f"dump_hidden sha256 {info['dump_hidden_sha256']}")
    todo = ps.todo()
    if not todo:
        return info

    work = scratch / "llamacpp_inputs"
    work.mkdir(parents=True, exist_ok=True)
    lines = []
    for p in todo:
        tf = work / f"{p['id']}.txt"
        idf = work / f"{p['id']}.ids.json"
        tf.write_text(p["text"], encoding="utf-8")
        idf.write_text(json.dumps(ps.token_ids[p["id"]]))
        lines.append(f"{p['id']}\t{tf}\t{idf}")
    manifest = work / "manifest.tsv"
    manifest.write_text("\n".join(lines) + "\n")

    cmd = [str(binary), str(project_root / ps.task_cfg["gguf"]), str(manifest), str(ps.run_dir),
           str(int(lcfg["n_ctx"])), "1" if lcfg["flash_attn"] else "0", str(int(lcfg["n_gpu_layers"]))]
    log = ps.run_dir / "dump_hidden.log"
    logger.info("$ " + " ".join(cmd))
    t = time.perf_counter()
    with open(log, "a") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    info["dump_hidden_wall_s"] = time.perf_counter() - t
    info["dump_hidden_rc"] = rc
    tail = log.read_text(errors="replace")[-3000:]
    logger.info(f"dump_hidden rc={rc}; log tail:\n{tail}")

    n_layers = ps.n_layers
    sources = {EMB: "cb_eval inp_embd", FINAL_NORM: "embeddings API (result_norm), pass A",
               FINAL_NORM_CB: "cb_eval result_norm, pass B", LAST_LOGITS: "llama_get_logits_ith(T-1), pass A"}
    for k in range(1, n_layers + 1):
        sources[pre_layer(k)] = f"cb_eval l_out-{k - 1}"
    failure: str | None = None
    for p in tqdm(todo, desc=f"{ps.task} validate"):
        pid = p["id"]
        d = ps.prompt_dir(pid)
        dm_path = d / "dump_meta.json"
        if failure is not None or not dm_path.exists():
            failure = failure or f"dump_hidden exited with rc={rc} before finishing {pid}: {tail[-1000:]}"
            ps.write_error(pid, failure)
            continue
        dm = json.loads(dm_path.read_text())
        if dm["n_layer"] != n_layers or dm["n_embd"] != ps.hidden or dm["n_tokens"] != len(ps.token_ids[pid]):
            raise RuntimeError(f"{pid}: dump_meta {dm} inconsistent with config")
        llama_ids = json.loads((d / "llama_token_ids.json").read_text())
        seconds = {layer: dm["pass_b_decode_with_capture_s"] for layer in ps.layers}
        seconds[FINAL_NORM] = dm["pass_a_decode_s"]
        seconds[LAST_LOGITS] = dm["pass_a_decode_s"]
        ok = ps.write_ok(
            pid, None, ps.token_ids[pid], seconds, "prompt", sources,
            extra_token_info={"llama_token_ids": llama_ids,
                              "llama_tokenization_equal_hf": llama_ids == ps.token_ids[pid]},
            extra_meta={"dump_meta": dm},
        )
        if not ok:
            failure = f"arrays of {pid} failed validation (shape or non-finite); method stopped"
    if failure is not None:
        raise MethodFailure(failure)
    if rc != 0:
        raise MethodFailure(f"dump_hidden exited with rc={rc} after writing all prompts: {tail[-1000:]}")
    return info


# ── Main ──────────────────────────────────────────────────────────────────────


def load_prompts(project_root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Load prompts.json and verify its fixed sha256 and size."""
    path = project_root / cfg["prompts_file"]
    digest = sha256_file(path)
    if digest != cfg["prompts_sha256"]:
        raise ValueError(f"{path} sha256 {digest} != approved {cfg['prompts_sha256']}")
    prompts = json.loads(path.read_text(encoding="utf-8"))["prompts"]
    ids = [p["id"] for p in prompts]
    if len(prompts) != int(cfg["n_prompts"]) or len(set(ids)) != len(ids):
        raise ValueError(f"expected {cfg['n_prompts']} unique prompts, got {ids}")
    for p in prompts:
        if not isinstance(p.get("text"), str) or not p["text"]:
            raise ValueError(f"prompt {p.get('id')} has no text")
    return prompts


def tokenize_prompts(prompts: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[dict[str, list[int]], str]:
    """Tokenize every prompt with the HF tokenizer of the snapshot (raw text, no special tokens)."""
    import transformers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["hf_snapshot"], local_files_only=True)
    out: dict[str, list[int]] = {}
    limit = int(cfg["vllm"]["max_model_len"]) - 1  # vLLM extract needs room for one generated token
    for p in prompts:
        ids = tok(p["text"], add_special_tokens=bool(cfg["add_special_tokens"]))["input_ids"]
        if not ids:
            raise ValueError(f"prompt {p['id']} tokenized to zero tokens")
        if len(ids) > limit:
            raise ValueError(f"prompt {p['id']} has {len(ids)} tokens > {limit}; plan says no truncation")
        out[p["id"]] = [int(i) for i in ids]
    return out, transformers.__version__


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")
    args = parse_args()
    variant_key = args.task  # plan task id

    exp_dir = Path(__file__).resolve().parent
    cfg = load_config(exp_dir)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE must be 1 (set in run_slurm.sh)")
    scratch_env = os.environ.get("SCRATCH_DIR")
    if not scratch_env:
        raise RuntimeError("SCRATCH_DIR is not set (provided by scripts/slurm_entry.sh)")
    scratch = Path(scratch_env)
    if set(cfg["tasks"]) != set(TASK_IDS):
        raise ValueError(f"config tasks {sorted(cfg['tasks'])} != plan tasks {sorted(TASK_IDS)}")
    task_cfg = cfg["tasks"][args.task]

    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    canonical_dir = project_root / "outputs" / exp_name / variant_key
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, task=args.task,
                       seed=int(cfg["seed"]), git_commit=git_revision(project_root))
    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir: {run_dir} (canonical {canonical_dir}); scratch: {scratch}")

    prompts = load_prompts(project_root, cfg)
    token_ids, tf_version = tokenize_prompts(prompts, cfg)
    n_layers = int(cfg["num_hidden_layers"])
    hidden = int(cfg["hidden_size"])
    ps = PromptSet(prompts, token_ids, run_dir, args.task, task_cfg, n_layers, hidden)

    # Resume: when staging on scratch, reuse prompts already finished in the canonical dir.
    if run_dir.resolve() != canonical_dir.resolve() and canonical_dir.is_dir():
        for p in prompts:
            src = canonical_dir / p["id"]
            meta = src / PROMPT_META
            if meta.exists() and json.loads(meta.read_text())["status"] == "ok" and not ps.is_done(p["id"]):
                shutil.copytree(src, run_dir / p["id"], dirs_exist_ok=True)
    done_before = [p["id"] for p in prompts if ps.is_done(p["id"])]
    logger.info(f"prompts finished in earlier attempts: {len(done_before)} {done_before}")

    info: dict[str, Any] = {
        "task": args.task, "task_config": task_cfg, "seed": int(cfg["seed"]),
        "git_commit": git_revision(project_root),
        "experiment_py_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(exp_dir / "config.yml"),
        "prompts_sha256": cfg["prompts_sha256"], "hf_snapshot": cfg["hf_snapshot"],
        "hf_revision": cfg["hf_revision"], "tokenizer_transformers": tf_version,
        "add_special_tokens": bool(cfg["add_special_tokens"]),
        "n_tokens_per_prompt": {pid: len(v) for pid, v in token_ids.items()},
        "resumed_prompts": done_before, "numpy": np.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME"),
        "method_failure": None,
    }
    atomic_write_text(run_dir / "run_info.json", json.dumps(info, indent=1, default=str))

    engine = task_cfg["engine"]
    try:
        if engine == "hf":
            info["engine_info"] = run_hf(ps, cfg, logger)
        elif engine == "vllm":
            info["engine_info"] = run_vllm(ps, cfg, scratch, logger)
        elif engine == "llamacpp":
            info["engine_info"] = run_llamacpp(ps, cfg, project_root, scratch, logger)
        else:
            raise ValueError(f"unknown engine {engine}")
    except MethodFailure as e:
        # Plan decision rule: the method is recorded as unavailable with its error.
        logger.error(f"METHOD UNAVAILABLE: {e}\n{traceback.format_exc()}")
        info["method_failure"] = str(e)
        for p in ps.todo():
            meta = ps.prompt_dir(p["id"]) / PROMPT_META
            if not meta.exists():
                ps.write_error(p["id"], f"method failed: {e}")

    cap = ps.write_capability()
    info["n_prompts_ok"] = sum(ps.is_done(p["id"]) for p in prompts)
    atomic_write_text(run_dir / "run_info.json", json.dumps(info, indent=1, default=str))
    logger.info(f"capability: {cap}; prompts ok {info['n_prompts_ok']}/{len(prompts)}; "
                f"method_failure={info['method_failure']}")

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
