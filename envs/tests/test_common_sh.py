"""Tests for envs/tools/common.sh and the CLI argument guards.

Only pure helpers and fail-fast paths are exercised; nothing here builds an
image, contacts a scheduler or needs a GPU.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ENVS = Path(__file__).resolve().parents[1]
COMMON = ENVS / "tools" / "common.sh"


def _bash(script: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run a bash snippet with common.sh sourced."""
    return subprocess.run(
        ["bash", "-c", f"source '{COMMON}'; {script}"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **(env or {})},
        check=False,
    )


def test_cmake_archs_conversion() -> None:
    out = _bash("nlpenv_cmake_archs '8.0;8.9;9.0;12.0'")
    assert out.stdout.strip() == "80;89;90;120"


def test_cuda_arch_override_wins() -> None:
    out = _bash("nlpenv_cuda_arch_list aarch64", env={"NLPENV_CUDA_ARCHS": "9.0"})
    assert out.stdout.strip() == "9.0"


def test_cuda_arch_baseline_without_gpu() -> None:
    # PATH without nvidia-smi -> only the per-arch baseline, sorted numerically.
    out = _bash("nlpenv_cuda_arch_list aarch64")
    assert out.stdout.strip() == "9.0;12.0"
    out = _bash("nlpenv_cuda_arch_list x86_64")
    assert out.stdout.strip() == "8.0;8.6;8.9;9.0;12.0"


def test_unknown_arch_fails() -> None:
    out = _bash("nlpenv_cuda_arch_list riscv64")
    assert out.returncode != 0
    assert "no CUDA arch baseline" in out.stderr


def test_store_paths_follow_override(tmp_path: Path) -> None:
    out = _bash(
        "nlpenv_venv_path aarch64 serve-vllm; nlpenv_sif_path x86_64 base",
        env={"NLPENV_STORE": str(tmp_path)},
    )
    assert out.stdout.split() == [
        f"{tmp_path}/aarch64/venvs/serve-vllm",
        f"{tmp_path}/x86_64/sif/base.sif",
    ]


def test_unknown_profile_fails() -> None:
    out = _bash("nlpenv_require_profile nope")
    assert out.returncode != 0
    assert "unknown profile" in out.stderr


def test_every_profile_has_pyproject() -> None:
    out = _bash('printf "%s\\n" "${NLPENV_PROFILES[@]}"')
    for name in out.stdout.split():
        assert (ENVS / "profiles" / name / "pyproject.toml").is_file(), name


def test_miyabi_rejects_internal_queue_and_long_debug() -> None:
    script = ENVS / "tools" / "submit_miyabi.sh"
    out = subprocess.run(
        ["bash", str(script), "--group", "g0", "--queue", "small-g", "--", "base"],
        capture_output=True, text=True, check=False,
    )
    assert out.returncode != 0 and "internal routing queue" in out.stderr
    out = subprocess.run(
        ["bash", str(script), "--group", "g0", "--queue", "debug-g", "--walltime", "01:00:00", "--", "base"],
        capture_output=True, text=True, check=False,
    )
    assert out.returncode != 0 and "at most 00:30:00" in out.stderr


def test_flash_attn_archs_conversion() -> None:
    # flash-attn 2.8 only knows the families 80/90/100/120; minors collapse.
    assert _bash("nlpenv_flash_attn_archs '8.0;8.6;8.9;9.0;12.0'").stdout.strip() == "80;90;120"
    assert _bash("nlpenv_flash_attn_archs '9.0;12.0;12.1'").stdout.strip() == "90;120"
    out = _bash("nlpenv_flash_attn_archs '7.5;8.0'")
    assert out.returncode != 0 and "no kernels for compute capability 7.5" in out.stderr


def test_compile_jobs_bounded_by_memory_and_cpus() -> None:
    # 32 CPUs / 80000 MiB / 9 GiB per job -> 8 jobs x 4 nvcc threads = 32.
    assert _bash("nlpenv_compile_jobs 32 80000 9").stdout.split() == ["8", "4"]
    # Tiny memory still runs one job.
    assert _bash("nlpenv_compile_jobs 48 2048 9").stdout.split() == ["1", "4"]
    # Few CPUs cap the job count and leave one thread each.
    assert _bash("nlpenv_compile_jobs 2 80000 9").stdout.split() == ["2", "1"]
    out = _bash("nlpenv_compile_jobs 0 80000 9")
    assert out.returncode != 0


def test_job_mem_follows_slurm_allocation() -> None:
    out = _bash("nlpenv_job_mem_mib", env={"SLURM_MEM_PER_NODE": "1000"})
    assert out.stdout.strip() == "1000"


def test_no_gpu_hides_detected_gpu(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"
    smi.write_text("#!/bin/bash\necho 12.0\n")
    smi.chmod(0o755)
    path = f"{bin_dir}:/usr/bin:/bin"
    assert _bash("nlpenv_detect_gpu_caps", env={"PATH": path}).stdout.strip() == "12.0"
    out = _bash("nlpenv_detect_gpu_caps", env={"PATH": path, "NLPENV_NO_GPU": "1"})
    assert out.stdout.strip() == ""


def test_flash_profiles_are_profiles() -> None:
    out = _bash('printf "%s\\n" "${NLPENV_FLASH_PROFILES[@]}"; echo ---; printf "%s\\n" "${NLPENV_PROFILES[@]}"')
    flash, profiles = out.stdout.split("---")
    assert set(flash.split()) == {"train-hf", "allinone"}
    assert set(flash.split()) <= set(profiles.split())


def test_smoke_check_knows_every_profile() -> None:
    import ast

    tree = ast.parse((ENVS / "tools" / "smoke_check.py").read_text())
    keys: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "PROFILE_IMPORTS":
            assert isinstance(node.value, ast.Dict)
            keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    profiles = _bash('printf "%s\\n" "${NLPENV_PROFILES[@]}"').stdout.split()
    assert set(profiles) == keys


def _stub_bin(tmp_path: Path, fail_on: str, curl_ok: bool = True) -> Path:
    """Create stub apptainer/curl commands.

    apptainer appends its argv to <tmp>/apptainer.log and fails when argv
    contains fail_on; curl fails when curl_ok is False.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "apptainer").write_text(
        "#!/bin/bash\n"
        '[[ "$1" == --version ]] && { echo "apptainer version stub"; exit 0; }\n'
        f'printf "%s\\n" "$*" >> "{tmp_path}/apptainer.log"\n'
        f'[[ "$*" == *"{fail_on}"* ]] && {{ echo "stub failure" >&2; exit 3; }}\n'
        "exit 0\n"
    )
    (bin_dir / "curl").write_text(f"#!/bin/bash\nexit {0 if curl_ok else 7}\n")
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    return bin_dir


def _run_build(
    tmp_path: Path, fail_on: str, *targets: str, curl_ok: bool = True, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    bin_dir = _stub_bin(tmp_path, fail_on, curl_ok)
    store = tmp_path / "store"
    arch = subprocess.run(["uname", "-m"], capture_output=True, text=True, check=True).stdout.strip()
    sif = store / arch / "sif" / "base.sif"
    sif.parent.mkdir(parents=True)
    sif.write_text("stub")
    return subprocess.run(
        ["bash", str(ENVS / "tools" / "build.sh"), "--no-smoke", *targets],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "NLPENV_STORE": str(store),
            "HOME": str(tmp_path),
            **(extra_env or {}),
        },
        check=False,
    )


def test_with_flash_attn_applies_to_allinone(tmp_path: Path) -> None:
    out = _run_build(
        tmp_path, "uv sync", "--with-flash-attn", "profile:allinone", extra_env={"NLPENV_NO_GPU": "1"}
    )
    assert out.returncode != 0  # the stub fails uv sync after logging argv
    log = (tmp_path / "apptainer.log").read_text()
    assert "uv sync --group flash" in log
    assert "FLASH_ATTN_CUDA_ARCHS=" in log and "NVCC_THREADS=" in log
    # CPU-only jobs hide CUDA from the container.
    assert "CUDA_VISIBLE_DEVICES=" in log


def test_without_flash_attn_no_flash_group(tmp_path: Path) -> None:
    _run_build(tmp_path, "uv sync", "profile:allinone")
    log = (tmp_path / "apptainer.log").read_text()
    assert "--group flash" not in log and "FLASH_ATTN_CUDA_ARCHS" not in log


def test_smoke_target_needs_built_venv_and_no_network(tmp_path: Path) -> None:
    # curl fails: a smoke-only run must not depend on the network preflight.
    out = _run_build(tmp_path, "never", "smoke:allinone", curl_ok=False)
    assert out.returncode != 0
    text = out.stdout + out.stderr  # build.sh tees its log to stdout
    assert "is not built" in text and "no network access" not in text


def _stub_slurm(tmp_path: Path) -> Path:
    """Stub sinfo/scontrol/sbatch so submit_slurm.sh --dry-run can run anywhere."""
    bin_dir = tmp_path / "sbin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "sinfo").write_text(
        "#!/bin/bash\n"
        'case "$*" in\n'
        "  *%l*) echo infinite ;;\n"
        "  *%c*) echo 48 ;;\n"
        "  *%m*) echo 120000 ;;\n"
        "  *%N*) echo node01 ;;\n"
        "esac\n"
    )
    (bin_dir / "scontrol").write_text("#!/bin/bash\necho NodeName=node01 Arch=x86_64\n")
    (bin_dir / "sbatch").write_text("#!/bin/bash\necho stub-sbatch-should-not-run >&2; exit 9\n")
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    return bin_dir


def _submit(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    bin_dir = _stub_slurm(tmp_path)
    return subprocess.run(
        ["bash", str(ENVS / "tools" / "submit_slurm.sh"), *args],
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "NLPENV_STORE": str(tmp_path / "store"), "HOME": str(tmp_path)},
        check=False,
    )


def test_submit_gpus_zero_and_dependency(tmp_path: Path) -> None:
    out = _submit(
        tmp_path, "--dry-run", "--family", "fam", "--gpus", "0", "--cpus", "32", "--mem", "80000",
        "--dependency", "afterok:123", "--", "--with-flash-attn", "profile:allinone",
    )
    assert out.returncode == 0, out.stderr
    assert "#SBATCH --gres=gpu:0" in out.stdout
    assert "#SBATCH --dependency=afterok:123" in out.stdout
    assert "export NLPENV_NO_GPU=1" in out.stdout
    assert "#SBATCH --cpus-per-task=32" in out.stdout and "#SBATCH --mem=80000M" in out.stdout


def test_submit_default_one_gpu(tmp_path: Path) -> None:
    out = _submit(tmp_path, "--dry-run", "--family", "fam", "--", "smoke:train-hf")
    assert out.returncode == 0, out.stderr
    assert "#SBATCH --gres=gpu:1" in out.stdout and "export NLPENV_NO_GPU=0" in out.stdout
    assert "--dependency" not in out.stdout


def test_submit_rejects_bad_gpus_and_dependency(tmp_path: Path) -> None:
    out = _submit(tmp_path, "--gpus", "x", "--", "base")
    assert out.returncode != 0 and "--gpus must be" in out.stderr
    out = _submit(tmp_path, "--dependency", "now", "--", "base")
    assert out.returncode != 0 and "--dependency must look like" in out.stderr


def test_build_profile_failure_is_not_recorded(tmp_path: Path) -> None:
    # A failing "uv sync" must fail the target and leave no manifest behind.
    out = _run_build(tmp_path, "uv sync", "profile:base")
    assert out.returncode != 0
    assert not list((tmp_path / "store").glob("*/manifest/profile-base.json"))
    # Locks live in the store copy, never in the repository.
    assert list((tmp_path / "store").glob("*/projects/base/pyproject.toml"))
    assert not (ENVS / "profiles" / "base" / "uv.lock").exists()


def test_build_profiles_collects_failures(tmp_path: Path) -> None:
    out = _run_build(tmp_path, "bash -c", "profiles")
    assert out.returncode != 0
    failed = next(line for line in (out.stderr + out.stdout).splitlines() if "FAILED:" in line)
    assert "profile:base" in failed and "profile:serve-vllm" in failed and "profile:bio-scispacy" in failed


def test_build_requires_target() -> None:
    out = subprocess.run(
        ["bash", str(ENVS / "tools" / "build.sh")], capture_output=True, text=True, check=False
    )
    assert out.returncode != 0 and "no target" in out.stderr
