# envs: all-in-one NLP environments

One entry point (`envs/bin/nlpenv`) for every runtime the NLP projects use:
generation (vLLM, SGLang, llama.cpp, Strata), fast training (Hugging Face
stack, Unsloth) and data/evaluation work. Each stack keeps its own uv venv
because vLLM, SGLang and Unsloth pin different torch versions; all of them
share one CUDA base image.

The same files build natively on **x86_64** (David, creator, andre, preproc)
and **aarch64** (Grace, Miyabi-G). The architecture and GPU compute capability
are detected on the build node; nothing is cross-compiled.

## Environments

| name | purpose | key pins (checked 2026-10-17) |
|---|---|---|
| `base` | preprocessing, classic ML, HF inference/embedding | torch 2.14.1+cu130, transformers 5.20.1 |
| `serve-vllm` | latest vLLM generation | vllm 0.31.0, torch 2.13.0 (pinned by vllm), transformers <5.18 |
| `serve-vllm-0.26` | frozen stack of KG-with-pubmed / oss-llm-apps | vllm 0.26.0, torch 2.11.0, transformers 5.13.x |
| `serve-sglang` | SGLang runtime/server | sglang 0.5.22, torch 2.14.1, transformers 5.19.0 |
| `train-hf` | fast full FT / LoRA / RL | transformers 5.20.1, trl 1.16.0, peft 0.21.2, deepspeed 0.19.7, liger-kernel 0.8.4, kernels 0.17.2; optional flash-attn 2.8.3.post2 |
| `train-unsloth` | Unsloth LoRA/QLoRA | unsloth 2026.10.6, torch 2.14.1 |
| `llamacpp` | GGUF inference, quantization, conversion | llama.cpp v0.7.0 (620d222), CUDA build |
| `strata` | Qwen3.8-Flash-Next MoE engine | Strata v0.1.45 |

The exact resolved versions are in each `profiles/<name>/uv.lock` (one lock
covers both architectures).

## Layout

```
envs/
  bin/nlpenv              runtime entry point (auto-detects arch)
  base/base.def           CUDA 13.0 + cuDNN devel + uv base image
  llamacpp/llamacpp.def   llama.cpp built from source on top of base
  profiles/<name>/        uv project per environment (pyproject.toml, uv.lock)
  tools/common.sh         arch/GPU detection and paths
  tools/build.sh          builds on the current node (inside a job)
  tools/submit_slurm.sh   picks partitions and submits build.sh (Slurm)
  tools/submit_miyabi.sh  submits build.sh to Miyabi (PBS)
  tools/smoke_check.py    import + CUDA kernel checks per profile
  tests/                  pytest for the shell helpers
  .store/                 git-ignored build output
    <arch>/sif/           base.sif, llamacpp.sif
    <arch>/venvs/<name>/  profile venvs
    <arch>/python/        uv-managed CPython
    <arch>/manifest/      build records (host, commit, SHA-256, CUDA archs)
    <arch>/reports/       smoke-check results
    <arch>/cache/         Triton / inductor / vLLM / FlashInfer caches
    cache/uv, cache/hf    uv wheel cache, Hugging Face cache
    logs/<arch>/          build logs and generated job scripts
```

Set `NLPENV_STORE` to put `.store` elsewhere. `HF_HOME`, when already set,
is reused instead of `.store/cache/hf` so existing model downloads are shared.

## Building

Builds run on compute nodes. Never run them on a login node.

### Slurm (David cluster)

```bash
# Everything, for every architecture that can see this checkout
envs/tools/submit_slurm.sh -- all
# Preview the generated job script without submitting
envs/tools/submit_slurm.sh --dry-run -- all
# One profile; re-resolve its lock
envs/tools/submit_slurm.sh -- --relock profile:serve-vllm
# train-hf with the source-built flash-attn wheel (long)
envs/tools/submit_slurm.sh --time 08:00:00 -- --with-flash-attn profile:train-hf
```

How `submit_slurm.sh` decides where to run:

1. The checkout's storage zone selects node families that can see it.
   `/workspace/david01` is local to david01, so only x86_64 builds there.
   A checkout on `/workspace/filesrv01` (or `02`) reaches both
   `creator-o`/`creator-i` (x86_64) and `grace-o`/`grace-i` (aarch64).
2. The family's real CPU architecture is read from `scontrol show node`.
   A mismatch, or a family with no responding node, stops the submission.
3. The smallest `small/medium/large/x-large` partition whose MaxTime fits
   `--time` is used, with the whole node's CPUs and 90% of its memory and
   one GPU (for smoke checks). Override with `--family`, `--cpus`, `--mem`.

For aarch64 on Grace, clone this repository on filesrv01/02 and run
`submit_slurm.sh` there. Do not copy x86_64 venvs to ARM nodes.

### Miyabi (PBS)

Clone the repository under `/work/<group>/<user>/` on Miyabi, then on the
login node:

```bash
envs/tools/submit_miyabi.sh --group gg17 --queue short-g -- base profile:serve-vllm          # dry-run
envs/tools/submit_miyabi.sh --group gg17 --queue short-g --submit -- base profile:serve-vllm # qsub
```

- `*-g` / `*-mig` queues are aarch64 GH200, and `*-c` queues are x86_64 with
  no GPU. The job checks `uname -m` again and stops on a mismatch.
- Miyabi time is a hard budget. The script stays in dry-run until
  `--submit` is given. Before `qsub` it runs `miyabi_preflight.py`
  (read-only) and does not submit if that fails. Check `show_token` too.
- Images are gzip SquashFS (Apptainer's default). Miyabi's Apptainer 1.3.5
  cannot open zstd images.
- If an unprivileged `apptainer build` is refused, build.sh retries with
  `--fakeroot`. If compute nodes have no internet, the build stops at its
  network preflight.

### What build.sh does

- `base`: builds `base.sif` from `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04`.
  CUDA 13.0 matches the R580 driver.
- `profile:<name>`: inside `base.sif`, runs `uv lock` (only when the lock is
  missing or `--relock` is given) and then `uv sync --locked` into
  `.store/<arch>/venvs/<name>`. It then runs `smoke_check.py`, which imports
  the stack and runs bf16 matmul and SDPA on the GPU, checked against a CPU
  reference.
- `llamacpp`: compiles llama.cpp for the CUDA archs below and verifies that
  `llama-server --list-devices` sees CUDA.
- `strata`: clones the pinned tag and runs `./setup.sh --check`. It downloads
  no model.
- Every target writes a manifest with the host, git commit, image SHA-256,
  lock SHA-256 and CUDA archs.

CUDA architectures compiled (override with `NLPENV_CUDA_ARCHS="9.0;12.0"`):

| CPU arch | baseline | covers |
|---|---|---|
| x86_64 | 8.0;8.6;8.9;9.0;12.0 | Ampere, Ada, Hopper, RTX PRO 6000 Blackwell |
| aarch64 | 9.0;12.0 | GH200 (Miyabi-G), GB10 (Grace, sm_121 runs sm_120 code) |

The detected GPU's capability is always added to the baseline.

## Using an environment

```bash
envs/bin/nlpenv info                       # arch, GPU, build status
envs/bin/nlpenv list
envs/bin/nlpenv run serve-vllm -- vllm serve Qwen/Qwen3-8B --port 8000
envs/bin/nlpenv run serve-sglang -- python -m sglang.launch_server --model-path Qwen/Qwen3-8B
envs/bin/nlpenv run train-hf -- accelerate launch train.py
envs/bin/nlpenv run train-unsloth -- python finetune.py
envs/bin/nlpenv run llamacpp -- llama-server -m model.gguf -ngl 999 --port 8080
envs/bin/nlpenv run llamacpp -- convert_hf_to_gguf.py /path/to/hf_model --outfile model.gguf
envs/bin/nlpenv run base --bind /workspace/david01/ohto/data -- python analysis.py
envs/bin/nlpenv shell train-hf
```

- Containers start with `--cleanenv`. Only CUDA device selection, HF/W&B
  tokens, thread counts and scheduler IDs are forwarded. The checkout, the
  HF cache and any `--bind` paths are mounted.
- Paths outside the checkout must be bound with `--bind`.
- Inside an experiment's `run_slurm.sh`, prefix the command, for example
  `RUN_COMMAND="${NLPUTIL}/envs/bin/nlpenv run serve-vllm --bind ${PROJECT_ROOT} -- python ${PYTHON_PATH} ..."`.
  Switching an existing experiment to another environment changes its
  runtime, so treat it as a plan change for that experiment.

Training speed-ups available in `train-hf`:

- `attn_implementation="kernels-community/flash-attn2"` or `"sdpa"`: prebuilt
  attention kernels from the Kernels Hub, no compilation needed.
- `liger_kernel.transformers.apply_liger_kernel_to_<model>()`, or
  `use_liger_kernel=True` in TRL/Transformers training args: fused kernels
  that cut activation memory.
- DeepSpeed ZeRO via `accelerate config`; bitsandbytes 8-bit optimizers.

### Strata

```bash
envs/bin/nlpenv strata-setup --model IQ3_XXS --context 32768   # downloads ~70 GB
envs/bin/nlpenv run strata -- ./setup.sh --port 8080            # starts the server
```

- Linux prebuilt engines are x64 only. On aarch64, `strata-setup` adds
  `--build`.
- Models go to `.store/<arch>/strata-models`.
- Strata targets consumer GPUs (it keeps hot MoE experts in VRAM and the rest
  in RAM). Its installer is large and fast-moving, so the tag is pinned and
  the model download is a separate, explicit step.

## Updating versions

1. Edit the pins in `profiles/<name>/pyproject.toml`, or `NLPENV_*_DEFAULT`
   in `tools/common.sh` for CUDA, llama.cpp or Strata.
2. Rebuild with `--relock` (profiles) or `--rebuild` (images) on every
   architecture you use.
3. Commit the updated `uv.lock`.

Keep `serve-vllm-0.26` frozen: it reproduces existing results.

## Tests

```bash
pytest -q envs/tests        # login-node safe: no builds, scheduler or GPU
```
