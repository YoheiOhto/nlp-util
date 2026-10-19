# 0002 preprocess_gguf_quantize

## 目的
0001 で作った BF16 GGUF を、llama.cpp v0.7.0 の `llama-quantize` で Q8_0 と Q4_K_M に量子化する。
0003（速度）と 0004（内部表現）で、量子化が速さと内部表現にどう効くかを見るための入力を作る。

## 比べるもの
比較はしない。前処理フェーズ。2モデル × 2形式 = 4ジョブ。
imatrix（重要度行列）は使わない。llama.cpp の標準手順そのままの量子化にする。

```experiment-conditions
{
  "schema_version": 1,
  "hypothesis": {
    "objective": "Produce Q8_0 and Q4_K_M GGUF files from the 0001 BF16 GGUFs for the llama.cpp vs vLLM comparison.",
    "study_type": "exploratory",
    "hypotheses": ["Not a hypothesis test: a preprocessing step verified by exit status, llama.cpp load check and SHA-256."],
    "baseline": "The 0001 BF16 GGUF of the same model.",
    "decision_rules": {"success": "llama-quantize exits 0, the output loads in llama.cpp and its SHA-256 is recorded", "failure": "non-zero exit, truncated file or load error"},
    "non_goals": ["no imatrix calibration", "no other quantization types", "no inference benchmarks"]
  },
  "data": {
    "identity": {
      "qwen3-8b": "outputs/0001_20261019_preprocess_gguf_bf16/qwen3-8b/model-BF16.gguf (SHA-256 recorded by 0001 and re-verified here)",
      "qwen3-30b-a3b": "outputs/0001_20261019_preprocess_gguf_bf16/qwen3-30b-a3b/model-BF16.gguf (SHA-256 recorded by 0001 and re-verified here)"
    },
    "sample_size": 4,
    "sample_size_rationale": "2 models x 2 quantization types chosen by the user.",
    "inclusion_exclusion": {"included": "both 0001 outputs", "excluded": "GGUF files made before approval under envs/.store/bench/gguf/"},
    "labels": {"not_applicable": true, "reason": "Quantization has no labels."},
    "input_paths": ["outputs/0001_20261019_preprocess_gguf_bf16/qwen3-8b/model-BF16.gguf", "outputs/0001_20261019_preprocess_gguf_bf16/qwen3-30b-a3b/model-BF16.gguf"],
    "prior_access": "Pre-approval quantization by a subagent took 8B Q8_0 293 s / Q4_K_M 71 s and needed --nv because llama-quantize links libcuda; those files are not used."
  },
  "split": {"not_applicable": true, "reason": "No dataset split; quantization only."},
  "preprocessing": {
    "steps": ["verify the 0001 SHA-256 of model-BF16.gguf", "llama-quantize model-BF16.gguf <run_dir>/model-<TYPE>.gguf.tmp <TYPE> 16", "rename .tmp after exit 0", "sha256sum the output"],
    "missing_values": "A missing or mismatching input GGUF aborts the job.",
    "tokenization": "Tokenizer metadata is copied unchanged from the BF16 GGUF.",
    "truncation": "No truncation.",
    "retrieval": "No retrieval."
  },
  "model": {
    "checkpoint": "BF16 GGUFs of Qwen/Qwen3-8B and Qwen/Qwen3-30B-A3B-Instruct-2507 from 0001",
    "revision": "HF snapshots b968826d9c46dd6066d109eabc6255188de91218 (8B) and 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe (30B-A3B), via 0001",
    "tokenizer": "embedded GGUF tokenizer from 0001",
    "tokenizer_revision": "same as the 0001 GGUF",
    "trainable_components": ["no trainable components (quantization only)"],
    "initialization": "Weights from the 0001 BF16 GGUF."
  },
  "prompt": {"not_applicable": true, "reason": "No generation in this phase."},
  "hyperparameters": {
    "batch": {"not_applicable": true, "reason": "No training or inference."},
    "optimizer": {"not_applicable": true, "reason": "No training."},
    "schedule": {"not_applicable": true, "reason": "No training."},
    "training_length": {"not_applicable": true, "reason": "No training."},
    "stopping_selection": {"not_applicable": true, "reason": "No training."},
    "precision": "Source BF16; outputs Q8_0 (8-bit block) and Q4_K_M (4-bit k-quant, mixed).",
    "quantization": "llama.cpp v0.7.0 llama-quantize types Q8_0 and Q4_K_M, no imatrix, 16 threads.",
    "selection_rationale": "Q8_0 is the near-lossless and Q4_K_M the most common practical llama.cpp quantization; chosen by the user to include quantization effects."
  },
  "seeds": {"not_applicable": true, "reason": "Deterministic quantization without imatrix."},
  "evaluation": {
    "primary_metrics": ["quantization success (exit 0 and llama.cpp load)"],
    "secondary_metrics": ["wall time [s]", "output size [bytes]", "llama-quantize reported size reduction"],
    "aggregation": "Per model and type.",
    "uncertainty": "Not applicable: single deterministic run per task.",
    "multiple_comparisons": "Not applicable.",
    "denominator": "4 tasks.",
    "failure_handling": "A failed task is recorded and not used downstream.",
    "quality_checks": {"input_integrity": "0001 SHA-256 must match", "output_integrity": "SHA-256 recorded"},
    "independent_verification": "0003 and 0004 re-verify the recorded SHA-256 before use."
  },
  "variants": {
    "tasks": [
      {"id": "qwen3-8b__q8_0", "settings": {"model": "qwen3-8b", "type": "Q8_0"}},
      {"id": "qwen3-8b__q4_k_m", "settings": {"model": "qwen3-8b", "type": "Q4_K_M"}},
      {"id": "qwen3-30b-a3b__q8_0", "settings": {"model": "qwen3-30b-a3b", "type": "Q8_0"}},
      {"id": "qwen3-30b-a3b__q4_k_m", "settings": {"model": "qwen3-30b-a3b", "type": "Q4_K_M"}}
    ],
    "total_tasks": 4,
    "baseline_id": {"not_applicable": true, "reason": "No comparison in this phase."},
    "selection_rule": "All four tasks are required; no other types."
  },
  "resources": {
    "profile": "slurm",
    "location": "david01 (*-david01 partitions)",
    "per_job": {"gpus": "1 visible device is not required; job runs with --gres=gpu:0 but apptainer --nv so libcuda resolves", "cpus": 16, "memory": "20000M", "walltime": "01:00:00 (30B Q8_0 took 1003 s before)", "disk": "about 8.7/5.0 GB (8B) and 32.5/about 18 GB (30B)"},
    "total_limits": {"jobs": 4, "external_api": "not used", "miyabi_time": "not used"},
    "valid_until": "2026-10-25T23:59:00+09:00"
  },
  "environment": {
    "runtime": "Apptainer env/env.sif -> envs/.store/x86_64/sif/llamacpp.sif (sha256 982d236213c2bc2b7ab7160a15a07c00fe4566eea42d760d97e5e364580b1b4d, llama.cpp v0.7.0 commit 620d2225cc436ab0b001e3200e590e3b286ef9ab); .venv -> allinone venv for experiment.py",
    "dependencies": {"quantizer": "/opt/llama.cpp/bin/llama-quantize (static build, CUDA archs 80;86;89;90;120)", "orchestration": "allinone venv: python 3.12, vllm 0.31.0, torch 2.13.0+cu130"},
    "allowed_files": ["experiments/0002_20261019_preprocess_gguf_quantize/experiment.py", "experiments/0002_20261019_preprocess_gguf_quantize/config.yml", "experiments/0002_20261019_preprocess_gguf_quantize/run_slurm.sh", "experiments/0002_20261019_preprocess_gguf_quantize/plan.md"],
    "shared_library_impact": "None; lib/output_utils.py unchanged."
  },
  "outputs": {
    "locations": {"gguf": "outputs/0002_20261019_preprocess_gguf_quantize/<variant>/model-<TYPE>.gguf", "metadata": "completion.json and run metadata with input and output SHA-256", "logs": "logs/0002_20261019_preprocess_gguf_quantize/"},
    "preservation": "Keep until 0003-0005 are reviewed; deletion only with user approval.",
    "resume_policy": "completed guard skips finished variants; an interrupted variant restarts from scratch."
  },
  "debug_policy": {
    "allowed_repairs": ["fix paths, bind mounts, --nv usage or invocation", "raise memory or walltime (resource-only)"],
    "retry_allowed": true,
    "resume_allowed": true,
    "equivalence_requirements": "Same llama.cpp v0.7.0 llama-quantize, same types, no imatrix; anything else needs a new plan version."
  },
  "stop_conditions": {
    "completion": "All four variants completed with recorded SHA-256.",
    "budget_deadline": "Stop at 2026-10-25T23:59+09:00; no external API or Miyabi use.",
    "correctness_failure": "Stop if an input SHA-256 does not match 0001 or the output fails to load.",
    "consultation_triggers": "A type is unsupported for a model or quantization needs an imatrix."
  }
}
```

## 実行状況
