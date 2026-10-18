import ast
import json
import sys
import os
import re
from pathlib import Path

def get_latest_experiment() -> Path:
    exp_dir = Path("experiments")
    subdirs = [d for d in exp_dir.iterdir() if d.is_dir() and re.match(r"^\d{4}_", d.name)]
    if not subdirs:
        print("❌ Error: No experiments found in experiments/")
        sys.exit(1)
    return sorted(subdirs)[-1]

def check_config_experiment_match(exp_path: Path):
    config_file = exp_path / "config.yml"
    exp_file = exp_path / "experiment.py"
    
    if not config_file.exists() or not exp_file.exists():
        return True
    
    config_content = config_file.read_text(encoding="utf-8")
    exp_content = exp_file.read_text(encoding="utf-8")
    
    # Simple YAML key parser (top level keys)
    config_keys = set(re.findall(r"^([a-zA-Z_][a-zA-Z0-9_]*):", config_content, re.MULTILINE))
    
    # Find config references in Python
    config_ref_pattern = r"""config(?:\.get\(\s*|\[\s*)(['"])([a-zA-Z_][a-zA-Z0-9_]*)\1"""
    exp_refs = {match.group(2) for match in re.finditer(config_ref_pattern, exp_content)}

    # 1. Undefined references
    undefined = exp_refs - config_keys
    if undefined:
        print(f"🔴 Blocker: Keys referenced in experiment.py but missing in config.yml: {undefined}")
        return False

    return True

def _active_lines(content: str) -> str:
    """Drop full-line comments so bash-variable checks ignore commented-out example blocks."""
    return "\n".join(
        line for line in content.split("\n") if not line.strip().startswith("#")
    )

def check_grid_size(exp_path: Path):
    run_slurm = exp_path / "run_slurm.sh"
    if not run_slurm.exists():
        return True

    content = run_slurm.read_text(encoding="utf-8")
    active = _active_lines(content)

    # Check for GRID_VALUES (only in active, uncommented lines)
    grid_matches = re.findall(r"GRID_VALUES=\((.*?)\)", active, re.DOTALL)
    grid_args_present = bool(re.search(r"GRID_ARGS\s*=\s*\(", active))

    total_tasks = None
    if grid_matches:
        values_str = grid_matches[0]
        lines = [line.strip() for line in values_str.split("\n") if line.strip()]
        total_tasks = 1
        for line in lines:
            clean_line = re.sub(r"['\"]", "", line)
            elements = clean_line.split()
            if elements:
                total_tasks *= len(elements)
        if total_tasks > 50:
            print(f"🔴 Blocker: GRID total task count ({total_tasks}) exceeds safety limit of 50.")
            return False
        elif total_tasks > 20:
            print(f"🟡 Warning: GRID task count is {total_tasks}. Ensure resources are optimized.")

    # Use the last assignment (bash "last one wins" if RUN_MODE is set more than once).
    run_mode_matches = re.findall(r'RUN_MODE\s*=\s*["\']?(\w+)["\']?', active)
    run_mode = run_mode_matches[-1] if run_mode_matches else "single"

    # #SBATCH directives are comments to bash, so this must be checked against the
    # raw content (not `active`), but anchored to line-start so a double-commented
    # example ("# #SBATCH --array=...") isn't mistaken for an active directive.
    array_match = re.search(r"^\s*#SBATCH\s+--array=\d+-(\d+)", content, re.MULTILINE)

    if grid_args_present and run_mode not in ("array", "seq"):
        print("🔴 Blocker: GRID_ARGS is defined but RUN_MODE is not \"array\" or \"seq\" "
              "— it would silently run as a single config. Set RUN_MODE and, for array "
              "mode, an active #SBATCH --array=0-N line.")
        return False

    if run_mode == "array":
        if not array_match:
            print("🔴 Blocker: RUN_MODE=\"array\" but no active #SBATCH --array=0-N line found.")
            return False
        if total_tasks is not None:
            declared_max = int(array_match.group(1))
            if declared_max != total_tasks - 1:
                print(
                    f"🔴 Blocker: #SBATCH --array=0-{declared_max} does not match the "
                    f"GRID combination count ({total_tasks}; expected 0-{total_tasks - 1})."
                )
                return False

    return True

def check_direct_writes(exp_path: Path):
    exp_file = exp_path / "experiment.py"
    if not exp_file.exists():
        return True
    
    content = exp_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(content, filename=str(exp_file))
    except SyntaxError as exc:
        print(f"🔴 Blocker: Invalid Python syntax in experiment.py: {exc}")
        return False

    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else None
        )
        path = node.args[0]
        if (name in {"to_parquet", "to_csv", "open"}
                and isinstance(path, ast.Constant)
                and isinstance(path.value, str)
                and re.match(r"^(?:\./)?(?:outputs|data)/", path.value)):
            matches.append(ast.get_source_segment(content, node) or name)
    if matches:
        print("🔴 Blocker: Hardcoded direct write to outputs/ or data/ detected in experiment.py:")
        for m in matches:
            print(f"   {m}")
        return False
    return True

def check_manifest(exp_path: Path):
    manifest = exp_path / "experiment.json"
    if not manifest.exists():
        print("🔴 Blocker: experiment.json is missing.")
        return False
    try:
        data = json.loads(manifest.read_text())
        resources = data["resources"]
        assert data["scheduler"] in {"auto", "slurm", "pbs", "local"}
        assert int(resources["gpus"]) >= 0
        assert int(resources["cpus"]) > 0
        assert int(resources["memory_gb"]) > 0
        assert re.fullmatch(r"\d+:\d{2}:\d{2}", resources["time"])
    except (OSError, ValueError, KeyError, AssertionError, TypeError) as exc:
        print(f"🔴 Blocker: invalid experiment.json: {exc}")
        return False
    return True

def check_scripts(exp_path: Path):
    import subprocess
    ok = True
    for name in ("run_slurm.sh", "run_qsub.sh", "run_local.sh"):
        path = exp_path / name
        if not path.exists():
            print(f"🔴 Blocker: {name} is missing.")
            ok = False
            continue
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if result.returncode:
            print(f"🔴 Blocker: {name} has invalid shell syntax: {result.stderr.strip()}")
            ok = False
    return ok

def check_resource_alignment(exp_path: Path):
    try:
        resources = json.loads((exp_path / "experiment.json").read_text())["resources"]
        slurm = (exp_path / "run_slurm.sh").read_text()
        pbs = (exp_path / "run_qsub.sh").read_text()
    except (OSError, KeyError, json.JSONDecodeError):
        return False
    expected = (
        f"#SBATCH --gres=gpu:{resources['gpus']}",
        f"#SBATCH --cpus-per-task={resources['cpus']}",
        f"#SBATCH --mem={resources['memory_gb']}g",
        f"#SBATCH --time={resources['time']}",
        f"#PBS -l walltime={resources['time']}",
        f"ncpus={resources['cpus']}",
        f"mem={resources['memory_gb']}gb",
    )
    missing = [value for value in expected[:4] if value not in slurm]
    missing += [value for value in expected[4:] if value not in pbs]
    if missing:
        print(f"🔴 Blocker: scheduler resources differ from experiment.json: {missing}")
        print("   Run: make render exp=<id>")
        return False
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path", nargs="?", default=None,
                         help="対象実験ディレクトリ（省略時は最新実験）")
    args = parser.parse_args()

    if args.exp_path:
        exp_path = Path(args.exp_path)
        if not exp_path.exists():
            print(f"❌ Error: experiment path not found: {exp_path}")
            sys.exit(1)
    else:
        exp_path = get_latest_experiment()

    print(f"🔬 Preflight checking experiment: {exp_path.name}")

    ok = True
    ok &= check_config_experiment_match(exp_path)
    ok &= check_grid_size(exp_path)
    ok &= check_direct_writes(exp_path)
    if (exp_path / "experiment.json").exists():
        ok &= check_manifest(exp_path)
        ok &= check_scripts(exp_path)
        ok &= check_resource_alignment(exp_path)

    if ok:
        print("✅ preflight OK — runx 投入可")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
