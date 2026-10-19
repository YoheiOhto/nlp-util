# 0004 extract_hidden_states

## 目的
llama.cpp と vLLM で内部表現（hidden states）がどこまで取れるか、取れた値が HF transformers とどれだけ一致するかを確かめる。
この実験は取り出しだけを行い、比較の計算は 0005 で行う。

## 比べるもの
- 基準: HF transformers bf16（`output_hidden_states=True` と最終 RMSNorm 前の hook）。精度の目安として HF fp32 も取る。
- vLLM bf16: 全層を返す `extract_hidden_states`（norm 前、L0〜L36）と、最終層を返す pooling runner（`task="token_embed"`, `use_activation=False`、norm 後）の2通り。
- llama.cpp BF16 / Q8_0 / Q4_K_M: C API の `cb_eval` で `inp_embd`・`l_out-*`（norm 前）・`result_norm`（norm 後）・最後のトークンの logits を保存する自作ツール `dump_hidden`（`envs/bench/hidden/llamacpp_dump/dump_hidden.cpp` を実験ディレクトリに固定してビルド）。
- 全エンジンに HF tokenizer の token ID をそのまま入れる。llama.cpp 自身の tokenize が HF と一致するかも別に記録する。
- モデルは Qwen3-8B だけ（ユーザーの決定）。

```experiment-conditions
{
  "schema_version": 1,
  "hypothesis": {
    "objective": "Determine which hidden representations llama.cpp and vLLM can return for Qwen3-8B, and save them with an HF transformers reference for the fidelity comparison in 0005.",
    "study_type": "exploratory",
    "hypotheses": ["Exploratory capability and fidelity check; no directional hypothesis is tested."],
    "baseline": "HF transformers bf16 (task hf-bf16).",
    "decision_rules": {"capability": "an extraction method counts as available when it returns per-token arrays of shape (n_tokens, 4096) for all 16 prompts with token IDs equal to the HF reference", "unavailable": "methods that fail or return mismatched shapes are recorded as unavailable with the error"},
    "non_goals": ["no Qwen3-30B-A3B hidden states", "no training or probing on the representations", "no new packages such as llama-cpp-python"]
  },
  "data": {
    "identity": {
      "prompts": "experiments/0004_20261019_extract_hidden_states/inputs/prompts.json version 1, sha256 dcb137efa9dcb3eaf16d1aa699f802c0c8944587663aa2a8cb44885fce86a42b, 16 prompts (English/Japanese, short/long, general/biomedical/code), written for this benchmark",
      "hf_model": "Qwen/Qwen3-8B snapshot b968826d9c46dd6066d109eabc6255188de91218",
      "gguf": "outputs/0001_20261019_preprocess_gguf_bf16/qwen3-8b/model-BF16.gguf, outputs/0002_20261019_preprocess_gguf_quantize/qwen3-8b__q8_0/model-Q8_0.gguf, outputs/0002_20261019_preprocess_gguf_quantize/qwen3-8b__q4_k_m/model-Q4_K_M.gguf (SHA-256 verified)"
    },
    "sample_size": 16,
    "sample_size_rationale": "16 prompts covering languages, lengths and domains are enough for a capability check and per-token fidelity statistics (thousands of tokens in total).",
    "inclusion_exclusion": {"included": "all 16 prompts", "excluded": "no prompt is dropped; a prompt whose token IDs differ between engines is kept and flagged"},
    "labels": {"not_applicable": true, "reason": "No labels; representations only."},
    "input_paths": ["experiments/0004_20261019_extract_hidden_states/inputs/prompts.json", "/workspace/david01/ohto/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218", "outputs/0001_20261019_preprocess_gguf_bf16/qwen3-8b", "outputs/0002_20261019_preprocess_gguf_quantize"],
    "prior_access": "Prompts and extraction scripts were drafted by a subagent; no extraction has been run."
  },
  "split": {"not_applicable": true, "reason": "No training or evaluation split."},
  "preprocessing": {
    "steps": ["tokenize each prompt with the HF tokenizer (add_special_tokens=False, no chat template) and save token IDs", "feed the same token IDs to every engine", "llama.cpp: also tokenize with llama.cpp and record whether IDs match HF", "save arrays per prompt and per engine as float32 .npy as soon as each prompt is done"],
    "missing_values": "A failed prompt is saved as an error record, not skipped silently.",
    "tokenization": "HF tokenizer of snapshot b968826d9c46dd6066d109eabc6255188de91218, add_special_tokens=False, raw text without chat template.",
    "truncation": "No truncation; the longest prompt is far below the 4096-token context used for all engines.",
    "retrieval": "No retrieval."
  },
  "model": {
    "checkpoint": "Qwen/Qwen3-8B (HF safetensors for HF and vLLM; 0001/0002 GGUFs for llama.cpp)",
    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "tokenizer": "HF tokenizer of the same snapshot",
    "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "trainable_components": ["no trainable components (inference only)"],
    "initialization": "Pretrained weights as published."
  },
  "prompt": {
    "templates": {"format": "raw text, no chat template, no system prompt", "content": "the 16 texts in inputs/prompts.json"},
    "content_refs": ["experiments/0004_20261019_extract_hidden_states/inputs/prompts.json sha256 dcb137efa9dcb3eaf16d1aa699f802c0c8944587663aa2a8cb44885fce86a42b"],
    "generation_settings": {"hf": "single forward pass, no generation", "vllm_extract_hidden_states": "max_tokens 1, temperature 0 (required by the connector); only prompt positions are compared", "vllm_pooling": "encode only", "llamacpp": "single llama_decode over the prompt, no sampling"},
    "renderer": "token IDs passed directly (HF input_ids, vLLM TokensPrompt, llama_batch)"
  },
  "hyperparameters": {
    "batch": {"hf": "batch size 1, no padding", "vllm": "all 16 prompts in one LLM call, enable_chunked_prefill False, enable_prefix_caching False, max_model_len 4096", "llamacpp": "one prompt per llama_decode, n_ctx 4096, n_batch = n_ubatch = 4096"},
    "optimizer": {"not_applicable": true, "reason": "No training."},
    "schedule": {"not_applicable": true, "reason": "No training."},
    "training_length": {"not_applicable": true, "reason": "No training."},
    "stopping_selection": {"not_applicable": true, "reason": "No training."},
    "precision": "hf-bf16: torch_dtype bfloat16, attn_implementation sdpa; hf-fp32: float32, sdpa; vllm: dtype bfloat16; llamacpp: GGUF weight type per task, -fa on, all layers on GPU.",
    "quantization": "Only the llama.cpp Q8_0 and Q4_K_M tasks are quantized (0002 files).",
    "selection_rationale": "Settings remove known sources of difference unrelated to the engine itself (chunked prefill, prefix cache, padding, chat templates)."
  },
  "seeds": {"values": [0], "repetitions": 1, "determinism": "Forward passes are deterministic up to GPU kernel nondeterminism; torch.manual_seed(0) for HF/vLLM; a single run per task."},
  "evaluation": {
    "primary_metrics": ["capability per method: which layers (embedding, each block output, final normed) and which positions are returned"],
    "secondary_metrics": ["extraction throughput [prompt tokens/s] after warm-up", "token-ID agreement rate of llama.cpp tokenization vs HF", "fidelity metrics are computed in 0005"],
    "aggregation": "Per task over the 16 prompts.",
    "uncertainty": "Not applicable here; 0005 reports distributions over tokens.",
    "multiple_comparisons": "Not applicable.",
    "denominator": "16 prompts; tokens counted per prompt.",
    "failure_handling": "Per-prompt errors are saved and reported as unavailable, never as zero similarity.",
    "quality_checks": {"shape": "(n_tokens, 4096) per layer", "token_ids": "engine input IDs equal HF IDs", "finite": "no NaN/Inf", "norm_alignment": "pre-norm arrays are compared with pre-norm reference and normed with normed (documented per method)"},
    "independent_verification": "0005 re-loads the saved arrays and checks shapes and token IDs before computing metrics."
  },
  "variants": {
    "tasks": [
      {"id": "hf-bf16", "settings": {"engine": "hf", "dtype": "bfloat16", "method": "output_hidden_states + final pre-norm hook", "layers": "L0-L36 pre-norm and final normed"}},
      {"id": "hf-fp32", "settings": {"engine": "hf", "dtype": "float32", "method": "output_hidden_states + final pre-norm hook", "layers": "L0-L36 pre-norm and final normed"}},
      {"id": "vllm-bf16-hidden", "settings": {"engine": "vllm", "dtype": "bfloat16", "method": "extract_hidden_states with ExampleHiddenStatesConnector", "layers": "L0-L36 pre-norm"}},
      {"id": "vllm-bf16-pooling", "settings": {"engine": "vllm", "dtype": "bfloat16", "method": "runner pooling, task token_embed, use_activation False", "layers": "final normed"}},
      {"id": "llamacpp-bf16", "settings": {"engine": "llamacpp", "gguf": "BF16", "method": "dump_hidden cb_eval", "layers": "inp_embd, l_out-0..35 pre-norm, result_norm, last-token logits"}},
      {"id": "llamacpp-q8_0", "settings": {"engine": "llamacpp", "gguf": "Q8_0", "method": "dump_hidden cb_eval", "layers": "inp_embd, l_out-0..35 pre-norm, result_norm, last-token logits"}},
      {"id": "llamacpp-q4_k_m", "settings": {"engine": "llamacpp", "gguf": "Q4_K_M", "method": "dump_hidden cb_eval", "layers": "inp_embd, l_out-0..35 pre-norm, result_norm, last-token logits"}}
    ],
    "total_tasks": 7,
    "baseline_id": "hf-bf16",
    "selection_rule": "All 7 tasks; no further methods."
  },
  "resources": {
    "profile": "slurm",
    "location": "david01 (*-david01 partitions)",
    "per_job": {"gpus": 1, "cpus": 16, "memory": "40000M", "walltime": "01:00:00", "disk": "up to about 3 GB per task (37 layers x float32 x tokens)"},
    "total_limits": {"jobs": 7, "external_api": "not used", "miyabi_time": "not used"},
    "valid_until": "2026-10-25T23:59:00+09:00"
  },
  "environment": {
    "runtime": "Apptainer env/env.sif -> envs/.store/x86_64/sif/llamacpp.sif (sha256 982d236213c2bc2b7ab7160a15a07c00fe4566eea42d760d97e5e364580b1b4d; llama.cpp v0.7.0 commit 620d2225cc436ab0b001e3200e590e3b286ef9ab static install with headers); .venv -> envs/.store/x86_64/venvs/allinone",
    "dependencies": {"hf": "transformers 5.17.0, torch 2.13.0+cu130 (allinone)", "vllm": "0.31.0 (allinone)", "llamacpp": "dump_hidden built from experiments/0004_20261019_extract_hidden_states/llamacpp_dump/ against /opt/llama.cpp (cmake, linked statically); binary SHA-256 recorded", "ordering": "submit only after environment jobs 14480/14481 have finished"},
    "allowed_files": ["experiments/0004_20261019_extract_hidden_states/experiment.py", "experiments/0004_20261019_extract_hidden_states/config.yml", "experiments/0004_20261019_extract_hidden_states/run_slurm.sh", "experiments/0004_20261019_extract_hidden_states/plan.md", "experiments/0004_20261019_extract_hidden_states/inputs/prompts.json", "experiments/0004_20261019_extract_hidden_states/llamacpp_dump/dump_hidden.cpp", "experiments/0004_20261019_extract_hidden_states/llamacpp_dump/CMakeLists.txt"],
    "shared_library_impact": "None; lib/output_utils.py unchanged."
  },
  "outputs": {
    "locations": {"arrays": "outputs/0004_20261019_extract_hidden_states/<variant>/<prompt_id>/<layer>.npy", "token_ids": "outputs/0004_20261019_extract_hidden_states/<variant>/<prompt_id>/token_ids.json", "summary": "outputs/0004_20261019_extract_hidden_states/<variant>/capability.parquet (method, layer, available, shape, seconds)", "logs": "logs/0004_20261019_extract_hidden_states/"},
    "preservation": "Keep arrays until 0005 is reviewed; deletion only with user approval.",
    "resume_policy": "Per-prompt outputs are written immediately; a resumed task skips prompts whose arrays and token IDs already exist."
  },
  "debug_policy": {
    "allowed_repairs": ["fix API usage, tensor names, shapes, paths or the dump_hidden build", "raise memory or walltime (resource-only)"],
    "retry_allowed": true,
    "resume_allowed": true,
    "equivalence_requirements": "Same prompts, token IDs, model files, dtypes and batch settings; a method that cannot work as specified is recorded as unavailable instead of being replaced."
  },
  "stop_conditions": {
    "completion": "All 7 tasks finished with arrays or recorded errors for every prompt.",
    "budget_deadline": "Stop at 2026-10-25T23:59+09:00; no external API or Miyabi use.",
    "correctness_failure": "Stop a method if token IDs cannot be passed unchanged or arrays are non-finite.",
    "consultation_triggers": "A method needs a new package, a different engine version, or changed dtype/batch settings."
  }
}
```

## 実行状況

### 2026-10-19 承認
- ユーザーがチャットで承認: 「Issue #4 の plan 0001〜0005 の実装と実行を承認」。記録は `execution-approval.json`（期限 2026-10-25 23:59 JST）。

### 2026-10-19 実装（投入前）
- テンプレート層を runx 互換版（KG-with-pubmed e9a501e の scripts/templates/tools/Makefile）に差し替えた。理由: 先に入れた template 7aae6e9 の slurm_entry.sh は run_slurm.sh から source される前提で、承認チェック付きの runx と組み合わせられない。出力は scratch を経由せず outputs/ に直接書く。
- SIF は run_slurm.sh で `SIF_PATH=${PROJECT_ROOT}/env/env.sif`（-> llamacpp.sif）を明示する（ログイン環境の SIF_PATH を引き継がないため）。
- 資源は runx の DEFAULT_* で指定。partition と USR1 の余裕は runx が自動で決める。
- 実装はサブエージェントが行い、親が plan と照合した。plan の承認部分は未変更（gate の digest 一致を確認）。
- 0004: vllm-bf16-hidden タスクで最後のトークンの logits も語彙全体で保存する（`logprobs_mode=raw_logits`）。hidden states の値には影響しない追加の取得。`gpu_memory_utilization=0.80` は運用上の値。エンジン側の失敗は plan どおり unavailable として記録する。

### 2026-10-19 環境ジョブの取り消し
- ユーザーの指示で、環境構築ジョブ 14480/14481（allinone への flash-attn 追加）を取り消し、実験を優先する。allinone venv は flash-attn なしのまま（store 側 lock sha256 f248bfde…）変わらないため、plan の「測定中に venv を変えない」目的はそのまま満たされる。flash-attn の追加は実験の後に行う。
