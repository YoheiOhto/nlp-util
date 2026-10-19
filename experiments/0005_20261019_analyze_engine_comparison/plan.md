# 0005 analyze_engine_comparison

## 目的
0003（速度）と 0004（内部表現）の生データから、表と図を作る。
- 速度: モデル × 形式 × 負荷ごとに、3回の中央値と範囲。
- 内部表現: HF bf16 を基準にした、トークンごとのコサイン類似度と相対 L2 誤差（層ごと）。最後のトークンの logits の一致度も出す。
GPU は使わない。数値の再計算だけを行う。

## 比べるもの
- 速度: vLLM bf16 を基準に、vLLM fp8 と llama.cpp BF16/Q8_0/Q4_K_M を並べる（検定はしない）。
- 内部表現: 各方法 vs HF bf16。HF fp32 vs HF bf16 を「bf16 の丸めだけで生じる差」の目安として並べる。norm 前は norm 前どうし、norm 後は norm 後どうしで比べる。

```experiment-conditions
{
  "schema_version": 1,
  "hypothesis": {
    "objective": "Aggregate 0003 speed results and compute hidden-state fidelity of llama.cpp and vLLM against HF transformers from 0004 outputs.",
    "study_type": "exploratory",
    "hypotheses": ["Exploratory description; no directional hypothesis is tested."],
    "baseline": "Speed: vLLM bf16 per model. Fidelity: HF transformers bf16 (0004 task hf-bf16).",
    "decision_rules": {"reporting": "every task of 0003 and 0004 appears in the tables; missing or invalid ones are shown as incomplete or unavailable", "interpretation": "fidelity of an engine is described relative to the hf-fp32 vs hf-bf16 gap; no pass/fail threshold is applied"},
    "non_goals": ["no new measurements", "no statistical tests", "no thresholds chosen after seeing results"]
  },
  "data": {
    "identity": {"speed": "outputs/0003_20261019_eval_engine_speed/*/ (vllm bench JSON, vram.csv, llama_bench.json)", "hidden": "outputs/0004_20261019_extract_hidden_states/*/ (npy arrays, token_ids.json, capability.parquet)"},
    "sample_size": 37,
    "sample_size_rationale": "All 30 speed tasks and all 7 hidden-state tasks.",
    "inclusion_exclusion": {"included": "all tasks with completion.json", "excluded": "nothing is dropped; invalid repeats are shown with their reason"},
    "labels": {"not_applicable": true, "reason": "No labels."},
    "input_paths": ["outputs/0003_20261019_eval_engine_speed", "outputs/0004_20261019_extract_hidden_states"],
    "prior_access": "None of these outputs exist yet."
  },
  "split": {"not_applicable": true, "reason": "Descriptive analysis; no split."},
  "preprocessing": {
    "steps": ["verify completion.json and guard_skipped_at of every input variant", "speed: parse vllm bench JSON per workload, apply the 0003 validity rules (zero failed requests, 256 output tokens)", "hidden: load arrays, check shapes and token-ID equality with hf-bf16, align layer names (HF hidden_states[k] = output of block k-1 pre-norm for k=1..36, hidden_states[0] = embeddings; final normed separately)"],
    "missing_values": "Unavailable methods and failed prompts are reported as such and excluded from the metric denominators, with counts shown.",
    "tokenization": "Token IDs come from 0004; prompts whose IDs differ are excluded from per-token metrics and listed.",
    "truncation": "No truncation.",
    "retrieval": "No retrieval."
  },
  "model": {"not_applicable": true, "reason": "No model is run; analysis of saved outputs only."},
  "prompt": {"not_applicable": true, "reason": "No generation."},
  "hyperparameters": {"not_applicable": true, "reason": "No training or inference."},
  "seeds": {"not_applicable": true, "reason": "Deterministic computation on saved data."},
  "evaluation": {
    "primary_metrics": ["speed: median output throughput [tok/s], median TTFT [ms], median TPOT [ms] per (model, format, workload)", "fidelity: per-token cosine similarity and relative L2 error ||x - x_ref|| / ||x_ref|| per layer vs hf-bf16"],
    "secondary_metrics": ["speed: min-max range, p99 TTFT/TPOT, load time, peak VRAM, llama-bench pp/tg", "fidelity: statistics with and without the first token (attention-sink outliers)", "last-token logits vs hf-bf16: top-1 agreement, top-10 overlap, KL(p_ref || p)", "extraction throughput from 0004"],
    "aggregation": "Speed: median and min-max over valid repeats. Fidelity: mean, median, p99 and max over tokens pooled across prompts, per layer and method; logits metrics averaged over the 16 prompts.",
    "uncertainty": "Speed: min-max over 3 repeats. Fidelity: distribution summaries over tokens (no inference to a population).",
    "multiple_comparisons": "Not applicable: no tests.",
    "denominator": "Speed: completed requests per workload. Fidelity: tokens with matching IDs (count reported per method); logits: prompts.",
    "failure_handling": "Missing tasks are shown as incomplete/unavailable; never imputed.",
    "quality_checks": {"recompute": "aggregates computed from raw files, not from 0003/0004 summaries", "alignment": "pre-norm vs pre-norm and normed vs normed only", "sanity": "hf-bf16 vs itself yields cosine 1 and error 0"},
    "independent_verification": "A second script path recomputes cosine/rel-L2 for 3 random (prompt, layer) pairs with float64 numpy and asserts agreement to 1e-6."
  },
  "variants": {
    "tasks": [{"id": "all", "settings": {"inputs": "0003 and 0004 outputs", "outputs": "tables and figures"}}],
    "total_tasks": 1,
    "baseline_id": "all",
    "selection_rule": "Single aggregation task."
  },
  "resources": {
    "profile": "slurm",
    "location": "david01 (*-david01 partitions)",
    "per_job": {"gpus": 0, "cpus": 16, "memory": "40000M", "walltime": "01:00:00"},
    "total_limits": {"jobs": 1, "external_api": "not used", "miyabi_time": "not used"},
    "valid_until": "2026-10-25T23:59:00+09:00"
  },
  "environment": {
    "runtime": "Apptainer env/env.sif -> envs/.store/x86_64/sif/llamacpp.sif (sha256 982d236213c2bc2b7ab7160a15a07c00fe4566eea42d760d97e5e364580b1b4d); .venv -> envs/.store/x86_64/venvs/allinone",
    "dependencies": {"numpy": "2.3.5 (allinone venv)", "pandas": "3.0.6 (allinone venv)", "pyarrow": "26.0.0 (allinone venv)", "matplotlib": "3.11.2 (allinone venv)"},
    "allowed_files": ["experiments/0005_20261019_analyze_engine_comparison/experiment.py", "experiments/0005_20261019_analyze_engine_comparison/config.yml", "experiments/0005_20261019_analyze_engine_comparison/run_slurm.sh", "experiments/0005_20261019_analyze_engine_comparison/plan.md"],
    "shared_library_impact": "None; lib/output_utils.py unchanged."
  },
  "outputs": {
    "locations": {"tables": "outputs/0005_20261019_analyze_engine_comparison/all/speed_summary.parquet, fidelity_by_layer.parquet, logits_agreement.parquet, capability.parquet", "figures": "outputs/0005_20261019_analyze_engine_comparison/all/figures/*.png with the source tables and the plotting script revision", "logs": "logs/0005_20261019_analyze_engine_comparison/"},
    "preservation": "Keep all tables and figures; reviewed conclusion appended below the status heading.",
    "resume_policy": "Re-running after a fix requires user approval to clear the completed guard."
  },
  "debug_policy": {
    "allowed_repairs": ["fix parsing, alignment or plotting bugs", "raise memory or walltime (resource-only)"],
    "retry_allowed": true,
    "resume_allowed": true,
    "equivalence_requirements": "Metric definitions, baselines and aggregation stay as listed."
  },
  "stop_conditions": {
    "completion": "All tables and figures written and the independent recomputation check passed.",
    "budget_deadline": "Stop at 2026-10-25T23:59+09:00; no external API or Miyabi use.",
    "correctness_failure": "Stop if the sanity or independent recomputation checks fail.",
    "consultation_triggers": "Inputs are too incomplete to fill the planned tables, or a metric definition proves inapplicable (e.g. layer counts differ)."
  }
}
```

## 実行状況

### 2026-10-19 承認
- ユーザーがチャットで承認: 「Issue #4 の plan 0001〜0005 の実装と実行を承認」。記録は `execution-approval.json`（期限 2026-10-25 23:59 JST）。
