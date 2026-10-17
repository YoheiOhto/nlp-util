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


def _stub_bin(tmp_path: Path, fail_on: str) -> Path:
    """Create stub apptainer/curl commands; apptainer fails when argv contains fail_on."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "apptainer").write_text(
        "#!/bin/bash\n"
        '[[ "$1" == --version ]] && { echo "apptainer version stub"; exit 0; }\n'
        f'[[ "$*" == *"{fail_on}"* ]] && {{ echo "stub failure" >&2; exit 3; }}\n'
        "exit 0\n"
    )
    (bin_dir / "curl").write_text("#!/bin/bash\nexit 0\n")
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    return bin_dir


def _run_build(tmp_path: Path, fail_on: str, *targets: str) -> subprocess.CompletedProcess[str]:
    bin_dir = _stub_bin(tmp_path, fail_on)
    store = tmp_path / "store"
    arch = subprocess.run(["uname", "-m"], capture_output=True, text=True, check=True).stdout.strip()
    sif = store / arch / "sif" / "base.sif"
    sif.parent.mkdir(parents=True)
    sif.write_text("stub")
    return subprocess.run(
        ["bash", str(ENVS / "tools" / "build.sh"), "--no-smoke", *targets],
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "NLPENV_STORE": str(store), "HOME": str(tmp_path)},
        check=False,
    )


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
