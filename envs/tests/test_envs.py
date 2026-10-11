"""Static checks for the shared NLP environments (nlp-envs); no GPU, network or container needed.

Run from the repository root:  python3 -m unittest discover -s envs/tests
"""
from __future__ import annotations

import re
import subprocess
import tomllib
import unittest
from pathlib import Path

ENVS = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ENVS / "pyproject.toml").read_text(encoding="utf-8"))
LOCK = tomllib.loads((ENVS / "uv.lock").read_text(encoding="utf-8"))
ARCHES = ("x86_64", "aarch64")
ENV_NAMES = ("nlp", "llm", "train", "es")

# Packages that have no wheel for a CPU and are compiled from source while building (needs the
# compilers in containers/base.def). A new entry means a slower, riskier build: review it first.
SOURCE_BUILDS = {
    ("nmslib-metabrainz", "aarch64"),  # needed by scispacy 0.5.5; no aarch64 wheel on PyPI
}


def canonical(name: str) -> str:
    """Normalize a distribution name (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement(spec: str) -> tuple[str, str]:
    """Split a PEP 508 requirement into (canonical name, version specifier)."""
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*([^;]*)", spec)
    assert match, spec
    return canonical(match.group(1)), match.group(3).replace(" ", "")


def wheel_arches(package: dict[str, object]) -> set[str]:
    """CPUs that a locked package has a wheel for (pure-Python wheels count for both)."""
    found: set[str] = set()
    for wheel in package.get("wheels", []):  # type: ignore[union-attr]
        url = wheel["url"]
        if url.endswith("-none-any.whl"):
            return set(ARCHES)
        found |= {arch for arch in ARCHES if f"_{arch}.whl" in url}
    return found


class LockTest(unittest.TestCase):
    """uv.lock must be the resolution of pyproject.toml, for both CPUs."""

    def test_only_linux_x86_64_and_aarch64_are_resolved(self) -> None:
        expected = {f"sys_platform == 'linux' and platform_machine == '{arch}'" for arch in ARCHES}
        self.assertEqual(set(PYPROJECT["tool"]["uv"]["environments"]), expected)
        self.assertEqual(len(LOCK["supported-markers"]), 2)
        self.assertTrue(all("linux" in m for m in LOCK["supported-markers"]))

    def test_lock_was_made_from_this_pyproject(self) -> None:
        root = next(p for p in LOCK["package"] if p["name"] == PYPROJECT["project"]["name"])
        locked = {(canonical(r["name"]), r.get("specifier", "").replace(" ", ""), r.get("marker", ""))
                  for r in root["metadata"]["requires-dist"]}
        wanted = {requirement(s) + ("",) for s in PYPROJECT["project"]["dependencies"]}
        for extra, specs in PYPROJECT["project"]["optional-dependencies"].items():
            wanted |= {requirement(s) + (f"extra == '{extra}'",) for s in specs}
        self.assertEqual(locked, wanted, "pyproject.toml changed after uv lock; run uv lock again")

    def test_exact_pins_are_what_got_locked(self) -> None:
        versions = {canonical(p["name"]): p["version"] for p in LOCK["package"] if "version" in p}
        specs = list(PYPROJECT["project"]["dependencies"])
        for extra in PYPROJECT["project"]["optional-dependencies"].values():
            specs += extra
        for spec in specs:
            name, specifier = requirement(spec)
            if specifier.startswith("=="):
                self.assertEqual(versions[name], specifier[2:], name)

    def test_every_package_installs_on_both_cpus(self) -> None:
        source_builds: set[tuple[str, str]] = set()
        for package in LOCK["package"]:
            if package.get("source", {}).get("virtual") or package.get("source", {}).get("editable"):
                continue
            missing = set(ARCHES) - wheel_arches(package)
            for arch in missing:
                self.assertIn("sdist", package, f"{package['name']} has neither a wheel nor a source for {arch}")
                source_builds.add((canonical(package["name"]), arch))
        self.assertEqual(source_builds, SOURCE_BUILDS,
                         "packages compiled from source changed; review them and update SOURCE_BUILDS")

    def test_torch_is_built_for_cuda_13_on_both_cpus(self) -> None:
        names = {canonical(p["name"]) for p in LOCK["package"]}
        for library in ("nvidia-cudnn-cu13", "nvidia-nccl-cu13", "nvidia-cutlass-dsl-libs-cu13"):
            self.assertIn(library, names)
        # nvidia-cutlass-dsl always requires its -cu12 libs and adds -cu13 through its "cu13" extra
        # (FlashInfer asks for it); the matching one is picked at run time. Nothing else may be CUDA 12.
        self.assertEqual({n for n in names if n.endswith("-cu12")}, {"nvidia-cutlass-dsl-libs-cu12"},
                         "CUDA 12 libraries crept in")


class VersionTest(unittest.TestCase):
    """The version is written once and published images are named from it."""

    def test_version_is_calendar_based(self) -> None:
        self.assertRegex(PYPROJECT["project"]["version"], r"^20\d{2}\.(0[1-9]|1[0-2])\.\d+$")

    def test_changelog_and_readme_name_the_current_version(self) -> None:
        version = PYPROJECT["project"]["version"]
        changelog = (ENVS / "CHANGELOG.md").read_text(encoding="utf-8")
        first = re.search(r"^## (\S+)", changelog, re.M)
        self.assertIsNotNone(first)
        self.assertEqual(first.group(1), version, "add a CHANGELOG.md section for the new version first")
        readme = (ENVS / "README.md").read_text(encoding="utf-8")
        self.assertIn(version, readme)
        for env in ENV_NAMES:
            for arch in ARCHES:
                self.assertIn(f"{env}-{version}-{arch}.sif", readme)

    def test_build_settings_are_complete(self) -> None:
        settings = PYPROJECT["tool"]["nlp-envs"]
        self.assertEqual(set(settings), {"cuda-image-tag", "uv-version", "python-version", "es-version"})
        self.assertTrue(settings["cuda-image-tag"].startswith("13."), "torch from PyPI is built for CUDA 13")
        self.assertEqual(PYPROJECT["project"]["requires-python"], f"=={settings['python-version']}.*")


class ScriptTest(unittest.TestCase):
    """Shell scripts must at least parse, and the definitions must use the build arguments they get."""

    def test_shell_scripts_parse(self) -> None:
        for script in sorted(ENVS.rglob("*.sh")) + sorted(ENVS.glob("*.sbatch")):
            with self.subTest(script=script.name):
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_definitions_receive_every_build_argument(self) -> None:
        build = (ENVS / "build.sh").read_text(encoding="utf-8")
        passed = set(re.findall(r'--build-arg "([A-Z_]+)=', build))
        for definition in sorted((ENVS / "containers").glob("*.def")):
            with self.subTest(definition=definition.name):
                used = set(re.findall(r"\{\{\s*([A-Z_]+)\s*\}\}", definition.read_text(encoding="utf-8")))
                self.assertTrue(used <= passed, f"{definition.name} uses {sorted(used - passed)} that build.sh never passes")

    def test_smoke_test_compiles_and_knows_every_extra(self) -> None:
        source = (ENVS / "tests" / "smoke.py").read_text(encoding="utf-8")
        compile(source, "smoke.py", "exec")
        self.assertEqual(set(re.findall(r'^    "(\w+)": \[', source, re.M)) - {"nlp"},
                         set(PYPROJECT["project"]["optional-dependencies"]),
                         "smoke.py REQUIRED must list one entry per extra in pyproject.toml")

    def test_host_python_settings_are_dropped_before_apptainer(self) -> None:
        # apptainer passes the caller's environment in; a UV_CACHE_DIR outside the image broke build job 2745.
        for script in ("build.sh", "test.sh"):
            with self.subTest(script=script):
                source = (ENVS / script).read_text(encoding="utf-8")
                scrub = source.find("UV_*|PYTHON*|VIRTUAL_ENV|CONDA_*|PIP_*) unset")
                self.assertNotEqual(scrub, -1)
                self.assertLess(scrub, source.find("apptainer build" if script == "build.sh" else "apptainer exec"))

    def test_build_time_checks_do_not_need_the_uv_cache(self) -> None:
        for definition in sorted((ENVS / "containers").glob("*.def")):
            with self.subTest(definition=definition.name):
                test_section = definition.read_text(encoding="utf-8").partition("%test")[2]
                commands = re.findall(r"^\s*uv\s+(\S+)", test_section, re.M)
                self.assertTrue(set(commands) <= {"--version"}, f"%test runs uv {commands}")

    def test_published_images_are_never_overwritten(self) -> None:
        build = (ENVS / "build.sh").read_text(encoding="utf-8")
        self.assertIn('if [ -e "$target" ]; then', build)
        self.assertIn("never overwritten", build)
        self.assertIn("chmod 0444", build)


if __name__ == "__main__":
    unittest.main()
