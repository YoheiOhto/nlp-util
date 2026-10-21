# 0003 eval_engine_speed

## 目的
同じ GPU（RTX PRO 6000 Blackwell 96 GB）で、llama.cpp（llama-server）と vLLM（vllm serve）の生成速度を比べる。
どちらも OpenAI 互換サーバーとして起動し、同じクライアント（`vllm bench serve`）で同じ長さの入出力を流す。

## 比べるもの
- エンジンと形式: vLLM bf16 / vLLM fp8（オンライン量子化）/ llama.cpp BF16 / Q8_0 / Q4_K_M
- モデル: Qwen3-8B（密）、Qwen3-30B-A3B-Instruct-2507（MoE）
- 1ジョブ = モデル × 形式 × 繰り返し1回。サーバーは毎回起動し直す（ロード時間と VRAM も毎回測る）。
- 1ジョブの中で、決めた5種類の負荷（下の JSON の workloads）を順に流す。負荷は1つの測定セットとして扱い、可変次元にはしない（サーバー起動を負荷ごとに繰り返すと 150 回のロードになり、比較の本質が変わらないため）。
- 仮説検定はしない。探索的な記録で、数値は3回の中央値と範囲で示す。

```experiment-conditions
{
  "schema_version": 1,
  "hypothesis": {
    "objective": "Measure generation speed, latency, load time and peak VRAM of llama.cpp and vLLM serving the same two models on the same GPU.",
    "study_type": "exploratory",
    "hypotheses": ["Exploratory: no directional hypothesis is tested; results are descriptive medians with ranges."],
    "baseline": "vLLM bf16 (task qwen3-8b__vllm-bf16__r1 and its repeats) for each model.",
    "decision_rules": {"usable_result": "a (model, format, workload) cell is reported when at least 2 of 3 repeats completed with zero failed requests", "incomplete": "cells with fewer than 2 valid repeats are reported as incomplete, not as zero"},
    "non_goals": ["no output-quality evaluation (accuracy is out of scope)", "no tuning of engine flags beyond the fixed settings below", "no other engines, models or quantization types", "no multi-GPU"]
  },
  "data": {
    "identity": {
      "requests": "synthetic random token prompts from vllm bench serve --dataset-name random with --seed 0, tokenized by each model's HF tokenizer (snapshot revisions below)",
      "qwen3-8b_hf": "Qwen/Qwen3-8B snapshot b968826d9c46dd6066d109eabc6255188de91218",
      "qwen3-30b-a3b_hf": "Qwen/Qwen3-30B-A3B-Instruct-2507 snapshot 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
      "gguf": "outputs/0001_20261019_preprocess_gguf_bf16/<model>/model-BF16.gguf and outputs/0002_20261019_preprocess_gguf_quantize/<model>__<type>/model-<TYPE>.gguf, SHA-256 verified against their metadata"
    },
    "sample_size": 30,
    "sample_size_rationale": "2 models x 5 engine formats x 3 repeats; 3 repeats give a median and range for run-to-run noise.",
    "inclusion_exclusion": {"included": "all requests of each workload", "excluded": "4 warm-up requests before each workload are sent and discarded"},
    "labels": {"not_applicable": true, "reason": "Speed measurement has no labels."},
    "input_paths": ["/workspace/david01/ohto/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218", "/workspace/david01/ohto/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe", "outputs/0001_20261019_preprocess_gguf_bf16", "outputs/0002_20261019_preprocess_gguf_quantize"],
    "prior_access": "No speed result has been measured; GPU jobs submitted before approval were held and will be cancelled with user approval."
  },
  "split": {"not_applicable": true, "reason": "No dataset split; synthetic benchmark requests."},
  "preprocessing": {
    "steps": ["start the server for the task and wait for /health (vLLM) or /health ok (llama-server); record load time", "for each workload: send 4 warm-up requests, then run vllm bench serve and save its JSON", "sample nvidia-smi memory.used every 500 ms during the job; record peak", "stop the server and verify the process exited"],
    "missing_values": "Failed requests are counted and reported; a workload with any failed request is marked invalid for that repeat.",
    "tokenization": "Prompts are generated as random token IDs by the HF tokenizer of the model; engines retokenize the decoded text, so actual prompt lengths are taken from each server's usage counts.",
    "truncation": "No truncation: max lengths fit (vLLM --max-model-len 4096; llama-server -c 65536 shared by 64 slots with -kvu unified KV cache).",
    "retrieval": "No retrieval."
  },
  "model": {
    "checkpoint": "Qwen/Qwen3-8B and Qwen/Qwen3-30B-A3B-Instruct-2507 (HF safetensors for vLLM; 0001/0002 GGUFs for llama.cpp)",
    "revision": "b968826d9c46dd6066d109eabc6255188de91218 (8B); 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe (30B-A3B)",
    "tokenizer": "HF tokenizer of the same snapshots (vLLM and client); GGUF-embedded tokenizer (llama.cpp)",
    "tokenizer_revision": "same snapshot revisions",
    "trainable_components": ["no trainable components (inference only)"],
    "initialization": "Pretrained weights as published."
  },
  "prompt": {
    "templates": {"format": "raw completion requests to /v1/completions; no chat template", "content": "random token sequences (benchmark load, not natural language)"},
    "content_refs": ["vllm bench serve --dataset-name random --seed 0 (vllm 0.31.0)"],
    "generation_settings": {"temperature": 0, "max_tokens": 256, "ignore_eos": true, "top_p": 1.0, "stream": true},
    "renderer": "vllm bench serve --backend openai --endpoint /v1/completions against each server"
  },
  "hyperparameters": {
    "batch": {"vllm": "--max-num-seqs 64, continuous batching", "llamacpp": "-np 64 slots, continuous batching (-cb is default)"},
    "optimizer": {"not_applicable": true, "reason": "No training."},
    "schedule": {"not_applicable": true, "reason": "No training."},
    "training_length": {"not_applicable": true, "reason": "No training."},
    "stopping_selection": {"not_applicable": true, "reason": "No training."},
    "precision": "vLLM bf16 weights and activations (fp8 task: --quantization fp8 online W8A8); llama.cpp BF16/Q8_0/Q4_K_M weights with -fa on.",
    "quantization": "vllm-fp8: --quantization fp8; llama.cpp Q8_0 and Q4_K_M from 0002 (no imatrix).",
    "selection_rationale": "Engine flags fixed before measurement: vLLM --max-model-len 4096 --max-num-seqs 64 --gpu-memory-utilization 0.90 --no-enable-prefix-caching; llama-server -ngl 999 -fa on -c 65536 -np 64 -kvu --cache-ram 0 --no-cache-prompt --threads 16. Prefix caching is off in both so repeated random prompts do not get cache hits.",
    "additional_settings": {
      "workloads": [
        {"id": "w1_single_in512", "input_len": 512, "output_len": 256, "max_concurrency": 1, "num_prompts": 16},
        {"id": "w2_single_in2048", "input_len": 2048, "output_len": 256, "max_concurrency": 1, "num_prompts": 16},
        {"id": "w3_c8", "input_len": 512, "output_len": 256, "max_concurrency": 8, "num_prompts": 64},
        {"id": "w4_c32", "input_len": 512, "output_len": 256, "max_concurrency": 32, "num_prompts": 256},
        {"id": "w5_c64", "input_len": 512, "output_len": 256, "max_concurrency": 64, "num_prompts": 512}
      ],
      "concurrency_1": "w1_single_in512 doubles as the concurrency-1 point of the throughput curve",
      "llamacpp_reference": "llama-bench -m <gguf> -ngl 999 -fa 1 -p 512,2048 -n 256 -r 3 -o json, run once per llama.cpp task after the server workloads"
    }
  },
  "seeds": {"values": [0], "repetitions": 3, "determinism": "Request contents are identical across tasks via --seed 0; timing varies between repeats, which is why 3 repeats with server restarts are taken."},
  "evaluation": {
    "primary_metrics": ["output throughput [tok/s] per workload", "median TTFT [ms] (w1, w2)", "median TPOT [ms] and decode speed 1000/TPOT [tok/s] (w1)"],
    "secondary_metrics": ["p99 TTFT and p99 TPOT", "total token throughput [tok/s]", "prefill speed = actual prompt tokens / median TTFT (w1, w2; includes HTTP and first decode step)", "model load time [s]", "peak VRAM [MiB] (vLLM includes its pre-allocated KV cache)", "llama-bench pp512/pp2048/tg256 [tok/s]"],
    "aggregation": "Per (model, format, workload): median and min-max over the valid repeats.",
    "uncertainty": "Range (min-max) over 3 repeats; no confidence intervals because n=3.",
    "multiple_comparisons": "Not applicable: descriptive comparison, no tests.",
    "denominator": "Requests per workload as listed; throughput uses completed requests only; failed requests are reported separately.",
    "failure_handling": "A repeat with failed requests or a server crash is recorded with its error and excluded from that cell; the cell is incomplete if fewer than 2 valid repeats remain.",
    "quality_checks": {"length_check": "completed output tokens must equal 256 per request (ignore_eos); otherwise the workload is invalid", "gguf_integrity": "SHA-256 of each GGUF matches 0001/0002 metadata", "server_shutdown": "server process absent after the task"},
    "independent_verification": "0005 recomputes all aggregates from the raw vllm bench JSON files."
  },
  "variants": {
    "tasks": [
      {"id": "qwen3-8b__vllm-bf16__r1", "settings": {"model": "qwen3-8b", "engine": "vllm", "format": "bf16", "repeat": 1}},
      {"id": "qwen3-8b__vllm-bf16__r2", "settings": {"model": "qwen3-8b", "engine": "vllm", "format": "bf16", "repeat": 2}},
      {"id": "qwen3-8b__vllm-bf16__r3", "settings": {"model": "qwen3-8b", "engine": "vllm", "format": "bf16", "repeat": 3}},
      {"id": "qwen3-8b__vllm-fp8__r1", "settings": {"model": "qwen3-8b", "engine": "vllm", "format": "fp8", "repeat": 1}},
      {"id": "qwen3-8b__vllm-fp8__r2", "settings": {"model": "qwen3-8b", "engine": "vllm", "format": "fp8", "repeat": 2}},
      {"id": "qwen3-8b__vllm-fp8__r3", "settings": {"model": "qwen3-8b", "engine": "vllm", "format": "fp8", "repeat": 3}},
      {"id": "qwen3-8b__llamacpp-bf16__r1", "settings": {"model": "qwen3-8b", "engine": "llamacpp", "format": "BF16", "repeat": 1}},
      {"id": "qwen3-8b__llamacpp-bf16__r2", "settings": {"model": "qwen3-8b", "engine": "llamacpp", "format": "BF16", "repeat": 2}},
      {"id": "qwen3-8b__llamacpp-bf16__r3", "settings": {"model": "qwen3-8b", "engine": "llamacpp", "format": "BF16", "repeat": 3}},
      {"id": "qwen3-8b__llamacpp-q8_0__r1", "settings": {"model": "qwen3-8b", "engine": "llamacpp", "format": "Q8_0", "repeat": 1}},
      {"id": "qwen3-8b__llamacpp-q8_0__r2", "settings": {"model": "qwen3-8b", "engine": "llamacpp", "format": "Q8_0", "repeat": 2}},
      {"id": "qwen3-8b__llamacpp-q8_0__r3", "settings": {"model": "qwen3-8b", "engine": "llamacpp", "format": "Q8_0", "repeat": 3}},
      {"id": "qwen3-8b__llamacpp-q4_k_m__r1", "settings": {"model": "qwen3-8b", "engine": "llamacpp", "format": "Q4_K_M", "repeat": 1}},
      {"id": "qwen3-8b__llamacpp-q4_k_m__r2", "settings": {"model": "qwen3-8b", "engine": "llamacpp", "format": "Q4_K_M", "repeat": 2}},
      {"id": "qwen3-8b__llamacpp-q4_k_m__r3", "settings": {"model": "qwen3-8b", "engine": "llamacpp", "format": "Q4_K_M", "repeat": 3}},
      {"id": "qwen3-30b-a3b__vllm-bf16__r1", "settings": {"model": "qwen3-30b-a3b", "engine": "vllm", "format": "bf16", "repeat": 1}},
      {"id": "qwen3-30b-a3b__vllm-bf16__r2", "settings": {"model": "qwen3-30b-a3b", "engine": "vllm", "format": "bf16", "repeat": 2}},
      {"id": "qwen3-30b-a3b__vllm-bf16__r3", "settings": {"model": "qwen3-30b-a3b", "engine": "vllm", "format": "bf16", "repeat": 3}},
      {"id": "qwen3-30b-a3b__vllm-fp8__r1", "settings": {"model": "qwen3-30b-a3b", "engine": "vllm", "format": "fp8", "repeat": 1}},
      {"id": "qwen3-30b-a3b__vllm-fp8__r2", "settings": {"model": "qwen3-30b-a3b", "engine": "vllm", "format": "fp8", "repeat": 2}},
      {"id": "qwen3-30b-a3b__vllm-fp8__r3", "settings": {"model": "qwen3-30b-a3b", "engine": "vllm", "format": "fp8", "repeat": 3}},
      {"id": "qwen3-30b-a3b__llamacpp-bf16__r1", "settings": {"model": "qwen3-30b-a3b", "engine": "llamacpp", "format": "BF16", "repeat": 1}},
      {"id": "qwen3-30b-a3b__llamacpp-bf16__r2", "settings": {"model": "qwen3-30b-a3b", "engine": "llamacpp", "format": "BF16", "repeat": 2}},
      {"id": "qwen3-30b-a3b__llamacpp-bf16__r3", "settings": {"model": "qwen3-30b-a3b", "engine": "llamacpp", "format": "BF16", "repeat": 3}},
      {"id": "qwen3-30b-a3b__llamacpp-q8_0__r1", "settings": {"model": "qwen3-30b-a3b", "engine": "llamacpp", "format": "Q8_0", "repeat": 1}},
      {"id": "qwen3-30b-a3b__llamacpp-q8_0__r2", "settings": {"model": "qwen3-30b-a3b", "engine": "llamacpp", "format": "Q8_0", "repeat": 2}},
      {"id": "qwen3-30b-a3b__llamacpp-q8_0__r3", "settings": {"model": "qwen3-30b-a3b", "engine": "llamacpp", "format": "Q8_0", "repeat": 3}},
      {"id": "qwen3-30b-a3b__llamacpp-q4_k_m__r1", "settings": {"model": "qwen3-30b-a3b", "engine": "llamacpp", "format": "Q4_K_M", "repeat": 1}},
      {"id": "qwen3-30b-a3b__llamacpp-q4_k_m__r2", "settings": {"model": "qwen3-30b-a3b", "engine": "llamacpp", "format": "Q4_K_M", "repeat": 2}},
      {"id": "qwen3-30b-a3b__llamacpp-q4_k_m__r3", "settings": {"model": "qwen3-30b-a3b", "engine": "llamacpp", "format": "Q4_K_M", "repeat": 3}}
    ],
    "total_tasks": 30,
    "baseline_id": "qwen3-8b__vllm-bf16__r1",
    "selection_rule": "All 30 tasks; no follow-up flag search."
  },
  "resources": {
    "profile": "slurm",
    "location": "david01 (the only GPU node that sees this repository); x-large-david01 or the smallest *-david01 class that fits walltime",
    "per_job": {"gpus": 1, "cpus": 16, "memory": "40000M", "walltime": "02:00:00", "note": "the GPU is shared with other experiments by queueing; jobs run one at a time"},
    "total_limits": {"jobs": 30, "gpu_hours_estimate": "about 15-30 GPU-hours (operational estimate, not a budget)", "external_api": "not used", "miyabi_time": "not used"},
    "valid_until": "2026-10-25T23:59:00+09:00"
  },
  "environment": {
    "runtime": "Apptainer env/env.sif -> envs/.store/x86_64/sif/llamacpp.sif (sha256 982d236213c2bc2b7ab7160a15a07c00fe4566eea42d760d97e5e364580b1b4d; contains base CUDA 13.0.2 + llama.cpp v0.7.0 commit 620d2225cc436ab0b001e3200e590e3b286ef9ab); .venv -> envs/.store/x86_64/venvs/allinone",
    "dependencies": {"vllm": "0.31.0", "torch": "2.13.0+cu130", "transformers": "5.17.0", "flashinfer-python": "0.7.0.post1", "llama.cpp": "v0.7.0 (llama-server, llama-bench)", "allinone_lock": "store-side uv.lock sha256 f248bfdefdc0e3ae5c7e9e2673fede71a0145fd3b0a570c9123c12709b325fa7; the venv state (uv pip freeze) is saved with every run", "ordering": "submit only after environment jobs 14480/14481 (flash-attn add to allinone) have finished, so the venv does not change during measurements"},
    "allowed_files": ["experiments/0003_20261019_eval_engine_speed/experiment.py", "experiments/0003_20261019_eval_engine_speed/config.yml", "experiments/0003_20261019_eval_engine_speed/run_slurm.sh", "experiments/0003_20261019_eval_engine_speed/plan.md"],
    "shared_library_impact": "None; lib/output_utils.py unchanged."
  },
  "outputs": {
    "locations": {"raw": "outputs/0003_20261019_eval_engine_speed/<variant>/<workload>.json (vllm bench result files), vram.csv, server.log, llama_bench.json", "summary": "outputs/0003_20261019_eval_engine_speed/<variant>/metrics.parquet", "logs": "logs/0003_20261019_eval_engine_speed/"},
    "preservation": "Keep all raw files; results of every repeat are kept even when invalid.",
    "resume_policy": "Each workload result is written as soon as it finishes; a resumed task restarts the server and skips workloads whose JSON already exists and passed the length check."
  },
  "debug_policy": {
    "allowed_repairs": ["fix server start/stop, health waiting, ports, bind mounts or client invocation", "raise walltime or memory (resource-only)"],
    "retry_allowed": true,
    "resume_allowed": true,
    "equivalence_requirements": "Engine flags, workloads, generation settings and model files must stay as listed; any change needs a new plan version."
  },
  "stop_conditions": {
    "completion": "All 30 tasks finished (valid or recorded as failed).",
    "budget_deadline": "Stop at 2026-10-25T23:59+09:00; no external API or Miyabi use.",
    "correctness_failure": "Stop a model/format if outputs are not 256 tokens or the server crashes in all repeats; report instead of changing flags.",
    "consultation_triggers": "A flag must change to make an engine run (e.g. memory), a model does not fit, or vllm bench cannot drive llama-server."
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
- 0003: クライアントに `--save-detailed`（256トークン確認のため）、`--num-warmups 4`（plan の warm-up 4件）、`--top-p 1.0`（plan の generation_settings）、`--metadata`（再開時の取り違え防止）を付ける。vLLM の `--dtype bfloat16` は snapshot の dtype と同じ。llama-bench はサーバー停止後に実行する。

### 2026-10-19 環境ジョブの取り消し
- ユーザーの指示で、環境構築ジョブ 14480/14481（allinone への flash-attn 追加）を取り消し、実験を優先する。allinone venv は flash-attn なしのまま（store 側 lock sha256 f248bfde…）変わらないため、plan の「測定中に venv を変えない」目的はそのまま満たされる。flash-attn の追加は実験の後に行う。
