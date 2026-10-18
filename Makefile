.PHONY: \
	setup \
	build_env \
	uv_sync \
	create_exp \
	mark_fail \
	resume_exp \
	rename_exp \
	log_clean \
	failed_log_clean \
	clean_failed \
	check \
	preflight \
	estimate \
	render \
	report_daily \
	report_weekly \
	remote_plan \
	status \
	jupyter \
	review

# =========================================================
# 0. Initial setup
# =========================================================

setup:
	@bash tools/first_setup.sh

# =========================================================
# 0b. Build the Apptainer image (env/env.def → env/env.sif)
#     partition はここで渡す。tools/build_env.sh 側には焼き込まない。
# =========================================================

build_env:
	@if [ -z "$(p)" ]; then \
		echo "❌ Error: partition is required."; \
		echo "Usage: make build_env p=<partition>"; \
		exit 1; \
	fi

	@mkdir -p logs
	sbatch --partition=$(p) tools/build_env.sh

# =========================================================
# 1. Sync Python environment
# =========================================================

uv_sync:
	@if [ -z "$(p)" ]; then \
		echo "❌ Error: partition is required."; \
		echo "Usage:"; \
		echo "  make uv_sync p=<partition>                  # base のみ"; \
		echo "  make uv_sync p=<partition> extras=\"llm chem\""; \
		echo "  make uv_sync p=<partition> extras=all        # 全extra"; \
		echo "Available extras: llm, chem, nlp"; \
		exit 1; \
	fi

	@mkdir -p logs
	sbatch --partition=$(p) tools/uv_sync.sh $(extras)

# =========================================================
# 2. Create Experiment
# =========================================================

create_exp:
	@if [ -z "$(name)" ]; then \
		echo "❌ Error: name is required."; \
		echo "Usage: make create_exp name=<exp_name>"; \
		exit 1; \
	fi

	@bash tools/create_exp.sh "$(name)"

# =========================================================
# 3. Mark Experiment as FAILED
# =========================================================

mark_fail:
	@if [ -z "$(name)" ] || [ -z "$(reason)" ]; then \
		echo "❌ Error: name and reason are required."; \
		echo "Usage:"; \
		echo "  make mark_fail name=<exp_name_or_index> reason=<reason>"; \
		exit 1; \
	fi

	@bash tools/mark_failed.sh "$(name)" "$(reason)"

# =========================================================
# 4. Resume Experiment
# =========================================================

resume_exp:
	@if [ -z "$(name)" ]; then \
		echo "❌ Error: name is required."; \
		echo "Usage:"; \
		echo "  make resume_exp name=<exp_name_or_index>"; \
		echo "  make resume_exp name=<exp_name_or_index> suffix=<suffix>"; \
		exit 1; \
	fi

	@bash tools/resume_exp.sh "$(name)" "$(suffix)"

# =========================================================
# 5. Rename Experiment
# =========================================================

rename_exp:
	@if [ -z "$(name)" ] || [ -z "$(new)" ]; then \
		echo "❌ Error: name and new are required."; \
		echo "Usage:"; \
		echo "  make rename_exp name=<exp_name_or_index> new=<new_name>"; \
		exit 1; \
	fi

	@bash tools/rename_exp.sh "$(name)" "$(new)"

# =========================================================
# 5b. Review Experiment
# =========================================================

review:
	@exp_val="$(exp)"; \
	if [ -z "$$exp_val" ]; then exp_val="$(name)"; fi; \
	bash tools/review.sh "$$exp_val"

# =========================================================
# 6. Clean up log files into job subdirectories
#    {job_id}_{exp_name}.out        → logs/{exp_name}/{job_id}/slurm.out
#    watcher_{job_id}.log           → logs/{exp_name}/{job_id}/watcher.log
#    （watcher_*.log は旧watch_job.sh方式の名残。新規には生成されないが、
#      既存ログの後方互換整理のためにロジックは残す）
# =========================================================

log_clean:
	@root=$$(git rev-parse --show-toplevel); \
	find "$$root/logs" -maxdepth 2 -name "*_*.out" | while read f; do \
		base=$$(basename "$$f"); \
		job_id=$$(echo "$$base" | cut -d_ -f1); \
		exp_dir=$$(dirname "$$f"); \
		dest="$$exp_dir/$$job_id"; \
		mkdir -p "$$dest"; \
		mv "$$f" "$$dest/slurm.out" && echo "  $$base → $$job_id/slurm.out"; \
	done; \
	find "$$root/logs" -maxdepth 2 -name "watcher_*.log" | while read f; do \
		base=$$(basename "$$f"); \
		job_id=$$(echo "$$base" | sed 's/watcher_//; s/\.log//'); \
		exp_dir=$$(dirname "$$f"); \
		dest="$$exp_dir/$$job_id"; \
		mkdir -p "$$dest"; \
		mv "$$f" "$$dest/watcher.log" && echo "  $$base → $$job_id/watcher.log"; \
	done

# =========================================================
# 6b. Clean up failed experiment logs and job directories
#     - FAILED/TIMEOUT/OOM/NODE_FAIL/CANCELLED のジョブログを削除
#     - status が RUNNING のまま残ったゾンビログを削除
#       （スケジューラにジョブが存在しないことを確認できた場合のみ。
#         稼働中ジョブのログを消さないため）
#     - 対応する experiments/ が無い孤立ログを削除
# =========================================================

failed_log_clean:
	@bash tools/clean_failed_logs.sh

clean_failed: failed_log_clean


# =========================================================
# 7. Start Jupyter
# =========================================================

jupyter:
	@if [ -z "$(p)" ] || [ -z "$(mem)" ]; then \
		echo "❌ Error: partition and memory are required."; \
		echo "Usage:"; \
		echo "  make jupyter p=<partition> mem=<memory>"; \
		echo "Example:"; \
		echo "  make jupyter p=a6000 mem=32g"; \
		exit 1; \
	fi

	sbatch \
		--partition=$(p) \
		--mem=$(mem) \
		tools/start_jupyter.sh

# =========================================================
# 8. Preflight validation checks before runx
# =========================================================

check:
	@bash tools/code_check.sh

preflight:
	@python3 -m scripts.preflight_check

estimate:
	@python3 tools/resource_estimate.py --root . $(if $(exp),--exp $(exp),)

render:
	@if [ -z "$(exp)" ]; then echo "Usage: make render exp=<id-or-directory>"; exit 1; fi
	@python3 tools/render_resources.py $(exp)

report_daily:
	@python3 tools/report.py daily --root .

report_weekly:
	@python3 tools/report.py weekly --root .

remote_plan:
	@bash tools/remote_run.sh plan $(if $(exp),--exp=$(exp),)

status:
	@python3 tools/research_status.py --root .
