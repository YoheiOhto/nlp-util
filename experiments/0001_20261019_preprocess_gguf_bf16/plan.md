# 0001 preprocess_gguf_bf16

## 目的
llama.cpp と vLLM の比較（0003 速度、0004 内部表現）で使う GGUF の元を作る。
HF の safetensors（固定 snapshot）を、llama.cpp v0.7.0 の `convert_hf_to_gguf.py` で BF16 GGUF に変換する。
重みの形式を変えるだけで、精度を落とす処理はしない。量子化は 0002 で行う。

## 比べるもの
比較はしない。前処理フェーズ。2モデルをそれぞれ1ジョブで変換する。

## 備考
承認前に別経路で作った GGUF（`envs/.store/bench/gguf/`）は使わない。この実験の出力だけを下流の入力にする。

```experiment-conditions
{
  "schema_version": 1,
  "hypothesis": {
    "objective": "Produce BF16 GGUF files of two fixed HF checkpoints for the llama.cpp vs vLLM comparison (experiments 0003 and 0004).",
    "study_type": "exploratory",
    "hypotheses": ["Not a hypothesis test: a lossless format conversion step whose output is verified by file integrity and a llama.cpp load check."],
    "baseline": "The HF safetensors snapshot of each model is the source of truth.",
    "decision_rules": {"success": "convert_hf_to_gguf.py exits 0, the GGUF loads in llama-cli/llama-server metadata read (tensor count and type BF16 logged), and its SHA-256 is recorded", "failure": "non-zero exit, truncated file or load error"},
    "non_goals": ["no quantization (done in 0002)", "no inference benchmarks", "no reuse of GGUF files created before approval"]
  },
  "data": {
    "identity": {
      "qwen3-8b": "Qwen/Qwen3-8B snapshot b968826d9c46dd6066d109eabc6255188de91218 (HF cache)",
      "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B-Instruct-2507 snapshot 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe (HF cache)"
    },
    "sample_size": 2,
    "sample_size_rationale": "Two models chosen by the user: one dense (8B) and one MoE (30B-A3B).",
    "inclusion_exclusion": {"included": "all weight shards of each snapshot", "excluded": "no other models"},
    "labels": {"not_applicable": true, "reason": "Format conversion has no labels."},
    "input_paths": [
      "/workspace/david01/ohto/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
      "/workspace/david01/ohto/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
    ],
    "prior_access": "A subagent converted these snapshots before approval into envs/.store/bench/gguf/; those files are not used. Conversion times observed there (8B 283 s, 30B 1926 s) only inform walltime."
  },
  "split": {"not_applicable": true, "reason": "No train/validation/test data; format conversion only."},
  "preprocessing": {
    "steps": ["run /opt/llama.cpp/venv/bin/python /opt/llama.cpp/src/convert_hf_to_gguf.py <snapshot> --outtype bf16 --outfile <run_dir>/model-BF16.gguf.tmp", "rename .tmp to model-BF16.gguf only after exit 0", "sha256sum the GGUF and write it to the run metadata"],
    "missing_values": "Not applicable to weights; a missing shard aborts the job.",
    "tokenization": "Tokenizer is embedded into the GGUF by convert_hf_to_gguf.py from the same snapshot.",
    "truncation": "No truncation.",
    "retrieval": "No retrieval."
  },
  "model": {
    "checkpoint": "Qwen/Qwen3-8B and Qwen/Qwen3-30B-A3B-Instruct-2507",
    "revision": "b968826d9c46dd6066d109eabc6255188de91218 (8B); 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe (30B-A3B)",
    "tokenizer": "tokenizer.json of the same snapshots",
    "tokenizer_revision": "same snapshot revisions as the weights",
    "trainable_components": ["no trainable components (conversion only)"],
    "initialization": "Pretrained weights as published; no modification."
  },
  "prompt": {"not_applicable": true, "reason": "No generation in this phase."},
  "hyperparameters": {
    "batch": {"not_applicable": true, "reason": "No training or inference."},
    "optimizer": {"not_applicable": true, "reason": "No training."},
    "schedule": {"not_applicable": true, "reason": "No training."},
    "training_length": {"not_applicable": true, "reason": "No training."},
    "stopping_selection": {"not_applicable": true, "reason": "No training."},
    "precision": "BF16 output (--outtype bf16), identical to the checkpoint dtype.",
    "quantization": "None in this phase.",
    "selection_rationale": "BF16 keeps the published precision and is the common source for 0002 quantization and the BF16 comparison."
  },
  "seeds": {"not_applicable": true, "reason": "Deterministic conversion; no randomness."},
  "evaluation": {
    "primary_metrics": ["conversion success (exit 0 and llama.cpp metadata load)"],
    "secondary_metrics": ["conversion wall time [s]", "GGUF size [bytes]", "peak RSS [MB] from sacct"],
    "aggregation": "Per model; no aggregation across models.",
    "uncertainty": "Not applicable: single deterministic conversion per model.",
    "multiple_comparisons": "Not applicable: no statistical comparison.",
    "denominator": "2 models.",
    "failure_handling": "A failed model is recorded as failed and not passed downstream; no partial file is renamed to the final name.",
    "quality_checks": {"integrity": "SHA-256 recorded", "load_check": "llama.cpp reads the GGUF header and reports tensor count/type"},
    "independent_verification": "0002 and 0003 re-read the GGUF and verify the recorded SHA-256 before use."
  },
  "variants": {
    "tasks": [
      {"id": "qwen3-8b", "settings": {"model": "qwen3-8b", "snapshot": "b968826d9c46dd6066d109eabc6255188de91218", "outtype": "bf16"}},
      {"id": "qwen3-30b-a3b", "settings": {"model": "qwen3-30b-a3b", "snapshot": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe", "outtype": "bf16"}}
    ],
    "total_tasks": 2,
    "baseline_id": {"not_applicable": true, "reason": "No comparison in this phase."},
    "selection_rule": "Both tasks are required; no extra models."
  },
  "resources": {
    "profile": "slurm",
    "location": "david01 (*-david01 partitions); repository and HF cache are local to david01",
    "per_job": {"gpus": 0, "cpus": 16, "memory": "40000M", "walltime": "02:00:00 (30B conversion took 1926 s before)", "disk": "about 16.4 GB (8B) and 61.1 GB (30B) under outputs/"},
    "total_limits": {"jobs": 2, "external_api": "not used", "miyabi_time": "not used"},
    "valid_until": "2026-10-25T23:59:00+09:00"
  },
  "environment": {
    "runtime": "Apptainer env/env.sif -> envs/.store/x86_64/sif/llamacpp.sif (sha256 982d236213c2bc2b7ab7160a15a07c00fe4566eea42d760d97e5e364580b1b4d, llama.cpp v0.7.0 commit 620d2225cc436ab0b001e3200e590e3b286ef9ab, CUDA 13.0.2 base); .venv -> envs/.store/x86_64/venvs/allinone for experiment.py",
    "dependencies": {"conversion": "/opt/llama.cpp/venv inside llamacpp.sif (requirements-convert_hf_to_gguf.txt of llama.cpp v0.7.0)", "orchestration": "allinone venv: python 3.12, vllm 0.31.0, torch 2.13.0+cu130, transformers 5.17.0"},
    "allowed_files": ["experiments/0001_20261019_preprocess_gguf_bf16/experiment.py", "experiments/0001_20261019_preprocess_gguf_bf16/config.yml", "experiments/0001_20261019_preprocess_gguf_bf16/run_slurm.sh", "experiments/0001_20261019_preprocess_gguf_bf16/plan.md"],
    "shared_library_impact": "None; uses lib/output_utils.py unchanged."
  },
  "outputs": {
    "locations": {"gguf": "outputs/0001_20261019_preprocess_gguf_bf16/<variant>/model-BF16.gguf", "metadata": "outputs/0001_20261019_preprocess_gguf_bf16/<variant>/completion.json and run metadata with sha256", "logs": "logs/0001_20261019_preprocess_gguf_bf16/"},
    "preservation": "Keep GGUF files until 0002-0005 are reviewed; deletion only with user approval.",
    "resume_policy": "completed guard skips finished variants; an interrupted variant restarts the conversion from scratch (tmp file is overwritten)."
  },
  "debug_policy": {
    "allowed_repairs": ["fix paths, bind mounts or invocation of the converter", "raise memory or walltime (resource-only)"],
    "retry_allowed": true,
    "resume_allowed": true,
    "equivalence_requirements": "Output must remain a BF16 GGUF of the same snapshot converted by llama.cpp v0.7.0; changing the converter version or outtype needs a new plan version."
  },
  "stop_conditions": {
    "completion": "Both variants completed with recorded SHA-256 and load check.",
    "budget_deadline": "Stop at 2026-10-25T23:59+09:00; no external API or Miyabi use.",
    "correctness_failure": "Stop if the converter reports unsupported tensors or the load check fails.",
    "consultation_triggers": "Converter needs a different llama.cpp version, or a snapshot shard is missing or corrupt."
  }
}
```

## 実行状況

### 2026-10-19 承認
- ユーザーがチャットで承認: 「Issue #4 の plan 0001〜0005 の実装と実行を承認」。記録は `execution-approval.json`（期限 2026-10-25 23:59 JST）。
