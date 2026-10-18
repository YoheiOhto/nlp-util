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
| `base` | preprocessing, classic ML, HF inference/embedding | torch 2.14.1+cu130, transformers 5.20.1, spacy 3.8 (numpy 2) |
| `bio-scispacy` | biomedical spaCy (scispacy), CPU only | scispacy 0.6.2, spacy <3.9, numpy <2 (cannot share a venv with numpy 2) |
| `serve-vllm` | latest vLLM generation and benchmarks | vllm 0.31.0, torch 2.13.0 (pinned by vllm), transformers <5.18, lm-eval 0.4.13 |
| `serve-vllm-0.26` | frozen stack of KG-with-pubmed / oss-llm-apps | vllm 0.26.0, torch 2.11.0, transformers 5.13.x |
| `serve-sglang` | SGLang runtime/server | sglang 0.5.22, torch 2.14.1, transformers 5.19.0 |
| `train-hf` | fast full FT / LoRA / RL | transformers 5.20.1, trl 1.16.0, peft 0.21.2, deepspeed 0.19.7, liger-kernel 0.8.4, kernels 0.17.2; optional flash-attn 2.8.3.post2 |
| `train-unsloth` | Unsloth LoRA/QLoRA | unsloth 2026.10.6, torch 2.14.1 |
| `allinone` | generation + training + data stack in one venv (RL with vLLM rollouts, eval after training) | vllm 0.31.0, torch 2.13.0, transformers 5.17.0, flashinfer 0.7.0.post1, trl 1.15.0, peft 0.21.2, accelerate 1.15.0, deepspeed 0.19.7, liger-kernel 0.8.4, kernels 0.17.2, bitsandbytes 0.50.2, unsloth 2026.10.6, lm-eval 0.4.13, datasets 4.8.5 (resolved 2026-10-18); optional flash-attn 2.8.3.post2. vLLM's torch pin and Unsloth's caps (transformers <=5.17.0, trl <=1.15.0, datasets <5) hold everything back; no scispacy (numpy <2) |
| `llamacpp` | GGUF inference, quantization, conversion | llama.cpp v0.7.0 (620d222), CUDA build |
| `strata` | Qwen3.8-Flash-Next MoE engine | Strata v0.1.45 |

No `uv.lock` is kept in the repository. Each build resolves the pins above
in `.store/<arch>/projects/<name>/` and keeps the resulting lock there as the
record of what was installed on that architecture.

## Layout

```
envs/
  bin/nlpenv              runtime entry point (auto-detects arch)
  base/base.def           CUDA 13.0 + cuDNN devel + uv base image
  llamacpp/llamacpp.def   llama.cpp built from source on top of base
  profiles/<name>/        uv project per environment (pyproject.toml only)
  tools/common.sh         arch/GPU detection and paths
  tools/build.sh          builds on the current node (inside a job)
  tools/submit_slurm.sh   picks partitions and submits build.sh (Slurm)
  tools/submit_miyabi.sh  submits build.sh to Miyabi (PBS)
  tools/smoke_check.py    import + CUDA kernel checks per profile
  tests/                  pytest for the shell helpers
  .store/                 git-ignored build output
    <arch>/sif/           base.sif, llamacpp.sif
    <arch>/venvs/<name>/  profile venvs
    <arch>/projects/<name>/ pyproject copy + resolved uv.lock of that build
    <arch>/python/        uv-managed CPython
    <arch>/manifest/      build records (host, commit, SHA-256, CUDA archs)
    <arch>/reports/       smoke-check results
    <arch>/cache/         Triton / inductor / vLLM / FlashInfer caches
    cache/uv              uv wheel cache
    logs/<arch>/          build logs and generated job scripts
```

Set `NLPENV_STORE` to put `.store` elsewhere. Models use `HF_HOME` when set,
else the standard `~/.cache/huggingface`, so existing downloads are shared.

## Building

Builds run on compute nodes. Never run them on a login node.

### Slurm (David cluster)

```bash
# Everything, for every architecture that can see this checkout
envs/tools/submit_slurm.sh -- all
# Preview the generated job script without submitting
envs/tools/submit_slurm.sh --dry-run -- all
# One profile, re-resolved from scratch
envs/tools/submit_slurm.sh -- --relock profile:serve-vllm
# Source-built flash-attn for train-hf and allinone (long, CPU only),
# then a short GPU job that re-runs their smoke checks after it succeeds
envs/tools/submit_slurm.sh --time 08:00:00 --gpus 0 --cpus 32 --mem 80000 \
    -- --with-flash-attn profile:train-hf profile:allinone
envs/tools/submit_slurm.sh --time 01:00:00 --gpus 1 --cpus 8 --mem 24000 \
    --dependency afterok:<compile job id> -- smoke:train-hf smoke:allinone
```

Options of `submit_slurm.sh` besides `--arch/--family/--time/--cpus/--mem`:

- `--gpus N` (default 1): GPUs requested with `--gres=gpu:N`. `--gpus 0`
  requests `--gres=gpu:0` and exports `NLPENV_NO_GPU=1`, so the job skips GPU
  detection, hides CUDA from its containers and runs the CPU part of the smoke
  checks only. Use it for long compile jobs so they do not hold the GPU that
  benchmark jobs need (david01 has one GPU, 48 CPUs and ~120 GB; a
  `--cpus 32 --mem 80000` compile leaves 16 CPUs and ~28 GB for others).
- `--dependency SPEC`: passed to `sbatch --dependency` (e.g. `afterok:12345`).
  build.sh also holds a per-arch lock, so an undepended smoke job simply
  waits for a running build.

How `submit_slurm.sh` decides where to run:

1. The checkout's storage zone selects node families that can see it.
   `/workspace/david01` is local to david01, so only x86_64 builds there.
   A checkout on `/workspace/filesrv01` (or `02`) reaches both
   `creator-o`/`creator-i` (x86_64) and `grace-o`/`grace-i` (aarch64).
2. The family's real CPU architecture is read from `scontrol show node`.
   A mismatch, or a family with no responding node, stops the submission.
3. The smallest `small/medium/large/x-large` partition whose MaxTime fits
   `--time` is used, with the whole node's CPUs and 90% of its memory and
   one GPU (for smoke checks). Override with `--family`, `--cpus`, `--mem`,
   `--gpus`.

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
- `profile:<name>`: copies `pyproject.toml` to `.store/<arch>/projects/<name>/`
  and runs `uv sync` there inside `base.sif`, installing into
  `.store/<arch>/venvs/<name>`. The store-side lock is reused while
  `pyproject.toml` is unchanged; a change or `--relock` re-resolves.
  It then runs `smoke_check.py`, which imports
  the stack and runs bf16 matmul and SDPA on the GPU, checked against a CPU
  reference. When flash-attn is installed it also compares
  `flash_attn_func` (causal, bf16) with an fp32 SDPA reference and checks its
  backward pass. For `train-hf` and `allinone` it runs one LoRA
  forward/backward/AdamW step on a tiny randomly initialized Llama model
  (2 layers, hidden 128; nothing is downloaded), with `sdpa` and, if
  installed, `flash_attention_2`. Without a GPU (e.g. `--gpus 0`) only the
  imports run, and GPU-only imports such as `unsloth` are recorded as skipped.
- `smoke:<name>`: re-runs only the smoke check of an existing venv, e.g. in a
  short GPU job after a CPU-only compile job. It needs no network.
- `--with-flash-attn`: adds the `flash` dependency group
  (`flash-attn==2.8.3.post2`) for `train-hf` and `allinone`, built from source
  (`FLASH_ATTENTION_FORCE_BUILD=TRUE`). flash-attn 2.8's `setup.py` ignores
  `TORCH_CUDA_ARCH_LIST` and reads `FLASH_ATTN_CUDA_ARCHS` (families 80, 90,
  100, 120), which build.sh derives from the arch list below (x86_64:
  `80;90;120`, aarch64: `90;120`; sm_86/sm_89 run the sm_80 kernels).
  `MAX_JOBS` is bounded by the job's memory (smaller of `MemAvailable` and
  `SLURM_MEM_PER_NODE`, ~9 GiB per nvcc process) and `NVCC_THREADS` fills
  the CPUs. A later `uv sync` of that profile without `--with-flash-attn`
  removes flash-attn again (uv sync is exact).
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

Put the entry point on `PATH` once (e.g. in `~/.bashrc`); `nlpenv` resolves
its checkout through the symlink, so it works from any directory:

```bash
mkdir -p ~/.local/bin
ln -sf /workspace/david01/ohto/1-projects/1-nlp/10-nlp-utils/github/nlp-util/envs/bin/nlpenv ~/.local/bin/nlpenv
```

```bash
envs/bin/nlpenv run serve-vllm -- lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-8B --tasks mmlu

envs/bin/nlpenv info                       # arch, GPU, build status
envs/bin/nlpenv list
envs/bin/nlpenv run serve-vllm -- vllm serve Qwen/Qwen3-8B --port 8000
envs/bin/nlpenv run serve-sglang -- python -m sglang.launch_server --model-path Qwen/Qwen3-8B
envs/bin/nlpenv run train-hf -- accelerate launch train.py
envs/bin/nlpenv run train-unsloth -- python finetune.py
envs/bin/nlpenv run allinone -- python grpo_with_vllm_rollouts.py
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

Training speed-ups available in `train-hf` and `allinone`:

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

Keep `serve-vllm-0.26` frozen: it reproduces existing results.

## Tests

```bash
pytest -q envs/tests        # login-node safe: no builds, scheduler or GPU
```
