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
    
    # Find config references in Python. Keep the two access forms separate:
    # combining them in one nested expression previously left an unmatched
    # parenthesis and made every preflight crash under Python 3.12.
    config_get_refs = re.findall(
        r"config\.get\(\s*['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]",
        exp_content,
    )
    config_item_refs = re.findall(
        r"config\[\s*['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]\s*\]",
        exp_content,
    )
    exp_refs = set(config_get_refs) | set(config_item_refs)
    
    # 1. Undefined references
    undefined = exp_refs - config_keys - {"upstream"} # upstream is standard
    if undefined:
        print(f"🔴 Blocker: Keys referenced in experiment.py but missing in config.yml: {undefined}")
        return False
        
    return True

def check_upstream_dependencies(exp_path: Path):
    config_file = exp_path / "config.yml"
    if not config_file.exists():
        return True
    
    content = config_file.read_text(encoding="utf-8")
    
    # Find upstream paths
    upstream_matches = re.findall(r"upstream:\s*\n((?:\s+.*\n?)+)", content)
    if not upstream_matches:
        return True
    
    lines = upstream_matches[0].split("\n")
    blocker = False
    for line in lines:
        match = re.search(r":\s*(outputs/\S+|data/\S+)", line)
        if match:
            path_str = match.group(1)
            path = Path(path_str)
            if not path.exists():
                print(f"🔴 Blocker: Upstream dependency path does not exist: {path_str}")
                blocker = True
            elif "outputs/" in path_str:
                # Check for completion.json
                run_dir = path if path.is_dir() else path.parent
                comp_json = run_dir / "completion.json"
                if comp_json.exists():
                    comp_content = comp_json.read_text(encoding="utf-8")
                    if '"status": "completed"' not in comp_content and '"status":"completed"' not in comp_content:
                        print(f"🔴 Blocker: Upstream experiment at {run_dir} is not completed (completion.json status is not completed)")
                        blocker = True
                else:
                    # Search up parent directories for completion.json
                    found_guard = False
                    p = run_dir
                    while p != Path(".") and len(p.parts) > 1:
                        if (p / "completion.json").exists():
                            comp_content = (p / "completion.json").read_text(encoding="utf-8")
                            if '"status": "completed"' in comp_content or '"status":"completed"' in comp_content:
                                found_guard = True
                            break
                        p = p.parent
                    if not found_guard:
                        print(f"🔴 Blocker: Upstream run directory {run_dir} does not contain completed completion.json")
                        blocker = True
    return not blocker

def check_grid_size(exp_path: Path):
    run_slurm = exp_path / "run_slurm.sh"
    if not run_slurm.exists():
        return True
    
    content = run_slurm.read_text(encoding="utf-8")
    # Check for GRID_VALUES
    grid_matches = re.findall(r"GRID_VALUES=\((.*?)\)", content, re.DOTALL)
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
    return True

def check_direct_writes(exp_path: Path):
    exp_file = exp_path / "experiment.py"
    if not exp_file.exists():
        return True
    
    content = exp_file.read_text(encoding="utf-8")
    matches = re.findall(r"(\b(?:to_parquet|to_csv|open)\(\s*['\"](?:./)?(?:outputs|data)/.*['\"])", content)
    if matches:
        print("🔴 Blocker: Hardcoded direct write to outputs/ or data/ detected in experiment.py:")
        for m in matches:
            print(f"   {m}")
        return False
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path", nargs="?", default=None,
                         help="対象実験ディレクトリ（省略時は最新実験）")
    parser.add_argument("--only-upstream", action="store_true",
                         help="upstream依存チェックのみ実行する（runx投入前hookから使用）")
    args = parser.parse_args()

    if args.exp_path:
        exp_path = Path(args.exp_path)
        if not exp_path.exists():
            print(f"❌ Error: experiment path not found: {exp_path}")
            sys.exit(1)
    else:
        exp_path = get_latest_experiment()

    if args.only_upstream:
        print(f"🔬 Upstream check: {exp_path.name}")
        if check_upstream_dependencies(exp_path):
            print("✅ upstream OK")
            sys.exit(0)
        sys.exit(1)

    print(f"🔬 Preflight checking experiment: {exp_path.name}")

    ok = True
    ok &= check_config_experiment_match(exp_path)
    ok &= check_upstream_dependencies(exp_path)
    ok &= check_grid_size(exp_path)
    ok &= check_direct_writes(exp_path)

    if ok:
        print("✅ preflight OK — runx 投入可")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
