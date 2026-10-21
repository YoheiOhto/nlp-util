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

---

> NOTE: the section below documents a second, independent environment system
> (Apptainer SIF images built by `build.sh` and `build.sbatch`, merged from origin/main).
> The section above documents `envs/bin/nlpenv` (uv venv profiles). Both live in `envs/`.
> The two README files conflicted (add/add) and both were kept.

# 言語班の共通環境 (nlp-envs)

**今の版: 2026.10.0** (2026-10-11) ／ 変更の記録: [CHANGELOG.md](CHANGELOG.md) ／ 作った SIF の記録: [RELEASES.md](RELEASES.md)

言語班のみんなが同じ版のパッケージで実験できるように、よく使う環境を4種類用意し、版つきで管理します。
x86 の計算機では Apptainer の SIF で使い、grace (ARM の CPU) では uv で同じ版を入れて使います (4.2)。
どれも **uv** で作り、uv で使えます。

> 2026-10-11 時点で、**x86 用の SIF 4つは micgm1 で作りました** (4つともテストを通過)。記録は [RELEASES.md](RELEASES.md)。
> grace 用の SIF は作りません (2026-10-11 に決めた)。

---

## 1. 何が解決するか

これまでは、プロジェクトごとに `pyproject.toml` と `uv.lock` を作っていたので、同じ言語班の中でも版がばらばらでした
(2026-10 の調べ: torch 2.9〜2.11、transformers 4.57〜5.12、vLLM 0.22〜0.23)。
組み合わせを各自で解くと、気づかないうちに古い版に戻ることもありました
(vLLM と同じ環境で `datasets` が 2020 年の 1.1.1 に戻っていたプロジェクトがあった)。

ここでは、組み合わせを一度だけ解いて (`uv.lock`)、x86 と ARM の両方で同じ版を使えるようにします。

## 2. 4つの環境

| 環境 | 使い道 | SIF (x86 の計算機用: creator・david・andre・hiegm5・preproc01・micgm1) | grace (ARM) |
| :--- | :--- | :--- | :--- |
| **nlp** | 前処理・BERT 系の学習と評価・データ処理 | `nlp-2026.10.0-x86_64.sif` | `uv sync --frozen` (4.2) |
| **llm** | vLLM での推論・LLM での判定・API | `llm-2026.10.0-x86_64.sif` | `uv sync --frozen --extra llm` |
| **train** | LoRA などの軽い学習 | `train-2026.10.0-x86_64.sif` | `uv sync --frozen --extra train` |
| **es** | Elasticsearch (検索のサーバー) | `es-2026.10.0-x86_64.sif` | 無し |

- SIF の名前は `<環境>-<版>-<CPU>.sif` です。**x86 用の SIF は grace では動きません**。`uname -m` で CPU を確かめます。
- grace では、同じ `uv.lock` から ARM 用の部品が入るので、x86 の SIF と同じ版になります。ただし grace の上で動くかは、まだ確かめていません (8章)。
- `llm` と `train` は、`nlp` の中身を全部含みます。

## 3. 中身と主な版

| | 版 | 入っている環境 |
| :--- | :--- | :--- |
| Python | 3.12 | 全部 |
| CUDA (コンテナの土台) | 13.0 (`nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`) | nlp・llm・train |
| torch | 2.13.0 (CUDA 13) | nlp・llm・train |
| transformers | 5.17.0 | nlp・llm・train |
| datasets / accelerate / tokenizers | 5.1.0 / 1.15.0 / 0.23.3 | nlp・llm・train |
| numpy / pandas / polars / duckdb | 2.3.5 / 3.0.6 / `uv.lock` のとおり | nlp・llm・train |
| spaCy | 3.8.16 (scispaCy は入っていない。8章) | nlp・llm・train |
| elasticsearch (クライアント) | 8.19.3 | nlp・llm・train |
| vLLM | 0.31.0 | llm |
| openai / anthropic / tiktoken | `uv.lock` のとおり | llm |
| peft / trl / bitsandbytes / wandb | 0.21.2 / 1.15.0 / 0.50.2 / `uv.lock` のとおり | train |
| Elasticsearch (サーバー) | 8.19.23 | es |

全部の版は [`uv.lock`](uv.lock) にあります (307 個)。何を入れるかは [`pyproject.toml`](pyproject.toml) に書いています。

## 4. 使い方

### 4.1 SIF をそのまま使う (一番かんたん)

```bash
SIF=<SIF を置いた場所>/llm-2026.10.0-$(uname -m).sif
apptainer exec --nv "$SIF" python my_script.py
```

- `--nv` は GPU を使うための指定です。コンテナの中では `python` が、この環境の Python を指します。
- ジョブの中で使います (ログインノードで重い計算はしない)。

### 4.2 SIF を使わずに、uv で同じ環境を作る

grace のように SIF を使いにくい機械や、手元で試すときです。CPU に合った版が自動で入ります。

```bash
git clone <nlp-util のリポジトリ> && cd nlp-util/envs
uv sync --frozen                  # nlp
uv sync --frozen --extra llm      # llm
uv sync --frozen --extra train    # train
```

- `--frozen` は「`uv.lock` のとおりに入れる。版を選び直さない」という意味です。
- できた `.venv` は数GB〜十数GB あります。ファイルサーバーの上だと読み込みが遅いので、なるべく 4.1 を使います。
- NVIDIA のドライバが 580 以上の機械で動きます (CUDA 13 のため)。preproc01 はドライバが入っていないので、GPU は使えません。

### 4.3 自分のプロジェクトで、パッケージを足して使う (uv)

共通の版をそのまま引き継ぎ、足りないものだけを足します。

```bash
cd my_project
cp <nlp-util>/envs/pyproject.toml <nlp-util>/envs/uv.lock .
# pyproject.toml の name をプロジェクトの名前に変え、[tool.nlp-envs] の表は消す
apptainer exec --nv llm-2026.10.0-$(uname -m).sif uv add <足すパッケージ>
apptainer exec --nv llm-2026.10.0-$(uname -m).sif uv sync --extra llm
```

- `uv add` は、`uv.lock` にある版をできるだけそのまま使って、足したものだけを解きます。共通の版から外れたら、`uv.lock` の差分に出ます。
- ちょっとだけ足したいときは、SIF の環境の上に小さな venv を重ねることもできます (SIF の中身は書き換えられないため):

  ```bash
  apptainer exec --nv "$SIF" uv venv --python /opt/nlp-envs/.venv/bin/python --system-site-packages .venv-extra
  apptainer exec --nv "$SIF" bash -c 'source .venv-extra/bin/activate && uv pip install <パッケージ> && python my_script.py'
  ```

### 4.4 Elasticsearch

```bash
bash envs/es/run_es.sh es-2026.10.0-$(uname -m).sif <データのフォルダ> 9200 4g
```

- ジョブの中で、その機械の中だけ (`127.0.0.1`) で動かします。Python からは `Elasticsearch("http://127.0.0.1:9200")` でつなぎます。
- **データのフォルダの注意**: 今までの SIF (8.13.2) で作ったデータを 8.19 で開くと、データが新しい形に書き換わり、8.13 では二度と開けなくなります。
  今のデータを使い続けるなら、先にコピーを取ってから開きます。

## 5. 版の決まり

- 版は **`年.月.番号`** です (例 `2026.10.0`)。同じ月に直したら番号を増やします (`2026.10.1`)。
- 版は [`pyproject.toml`](pyproject.toml) の `project.version` の1か所だけに書きます。SIF の名前・中のラベル・記録はここから作ります。
- **一度作った SIF は上書きしません。** 中身を変えるときは、必ず版を上げて別の名前で作ります。古い版は、使っている実験が無くなるまで残します。
- 作った SIF は [RELEASES.md](RELEASES.md) に、SHA-256 (中身が同じかを確かめる指紋) と一緒に記録します。
- Git のタグは `nlp-envs-<版>` (例 `nlp-envs-2026.10.0`) を、その版の SIF を全部作ってテストが通ったコミットに付けます。
- 実験の記録には、使った SIF の名前 (版と CPU 入り) を残します。

## 6. 更新のしかた

1. `pyproject.toml` を直す (パッケージを足す・版を変える)。版も上げ、[CHANGELOG.md](CHANGELOG.md) に何を変えたかを書く。
2. 組み合わせを解き直す (版の組み合わせを決めるだけで、パッケージは入れない):

   ```bash
   cd envs && apptainer exec <どれかの SIF> uv lock
   ```

3. 静的なテストを通す: `python3 -m unittest discover -s envs/tests` (リポジトリの一番上で)。
   - x86 と ARM のどちらかで入らない版、ソースからの組み立てが新しく必要になったパッケージ、`pyproject.toml` と `uv.lock` の食い違いを見つけます。
4. コミットしてから x86 用の SIF を作り (7章)、テストし、RELEASES.md に記録して、タグを付けます。

## 7. 作る

作るのは Slurm のジョブです。エージェント (Claude など) はジョブを出せない決まりなので、**人が出します**。
コミットしてから出します (変更が残っていると止まります)。

**x86 用 (micgm1 で)**: 作るジョブと、それが成功したら GPU で確かめるジョブを、1行でまとめて出します。

```bash
cd <nlp-util のフォルダ> && j=$(sbatch --parsable -p large-micgm1 envs/build.sbatch) && sbatch --kill-on-invalid-dep=yes -p small-micgm1 -d afterok:$j envs/test.sbatch && echo build=$j
```

- 4つ作るのに約 15 分でした (2026-10-11、micgm1。土台がまだ無いときはさらに数分)。作るジョブが失敗すると、確かめるジョブは自動で取り消されます。
- 一部だけ確かめ直すときは `sbatch -p small-micgm1 envs/test.sbatch --envs es` のように `--envs` を付けます。
  後ろにコメントを書かないでください (全角の空白の後ろの `#` はコメントにならず、引数になって止まります)。
- 置き場所は、既定でリポジトリの外の `../../images/` (`--out` で変えられます)。作業用の一時ファイルは `/scratch` があればそこ、無ければ置き場所の下の `.work/` です。

**grace 用**: SIF は作りません (2026-10-11 に決めた)。grace では 4.2 のとおり uv で入れます。
`build.sh` は ARM の機械で動かせば ARM 用を作るように書いてありますが、一度も試していません。

**作った後**: 各 SIF の横の `.sha256` の値を [RELEASES.md](RELEASES.md) に書き、`.test.log` が全部 PASS であることを確かめます。

## 8. 注意

- **ドライバ**: CUDA 13 なので、NVIDIA のドライバ 580 以上が必要です (2026-09 の機器一覧では、ドライバのある機械は全部 580 以上)。
- **GB10 (grace)**: torch は sm_120 の部品で GB10 (sm_121) を動かします。vLLM は 0.19 から GB10 に対応しています。
  grace の上ではまだ確かめていません。4.2 で入れたあと、GPU のジョブの中で `envs` から `uv run --frozen --extra llm python tests/smoke.py --env llm` を流すと確かめられます。
- **NVIDIA の CUTLASS のライブラリ**: `nvidia-cutlass-dsl` の決まりで、CUDA 12 用と 13 用の両方が入ります (実行時に合う方が使われる)。
- **Elasticsearch のデータ**: 4.4 の注意を読んでください。
- **scispaCy は入っていません**: どの版も numpy 1 が要り (0.5.5 は spaCy 3.7 を、0.6.2 は numpy<2 を求める)、numpy 2 が要る vLLM と同じ環境に入れられないためです。
  spaCy 3.7 と numpy 2 の組み合わせは、`uv lock` は通っても `import spacy` で止まります (`numpy.dtype size changed`)。
  text_preprocess_with_llm と claim-KG の `uv.lock` もこの組み合わせです (2026-10 時点ではどちらも spaCy を import していないので、表に出ていない)。
  scispaCy が要るときは、相談してください (numpy 1 の別の環境を作るなど)。

## 9. まだのこと

- タグ `nlp-envs-2026.10.0` は、この版が main に入ってから付けます。
- grace で uv で入れた環境が動くかは、まだ確かめていません (8章の GB10)。
- **ほかの人が使える置き場所**: 作った SIF をどこに置くかは未定です。案は、クラスタの各ノードの `/opt/lab/images` に配る (研究室のクラスタのキットの D-SIF-1)、
  またはファイルサーバーの共有フォルダ (filesrv01 と filesrv02 の両方。今はノードが片方しか見ないため) です。
- 言語班のほかの人のプロジェクトの環境は、まだ合わせていません (2026-10 は大戸のプロジェクト 4 つから作った)。
