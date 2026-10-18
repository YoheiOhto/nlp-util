#!/usr/bin/env python3
from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


output_utils = load("output_utils", ROOT / "lib/output_utils.py")
preflight = load("preflight", ROOT / "scripts/preflight_check.py")
resource_estimate = load("resource_estimate", ROOT / "tools/resource_estimate.py")
metadata = load("metadata", ROOT / "scripts/metadata.py")
render_resources = load("render_resources", ROOT / "tools/render_resources.py")
research_status = load("research_status", ROOT / "tools/research_status.py")


class TemplateTests(unittest.TestCase):
    def test_all_shell_files_parse(self):
        for path in ROOT.rglob("*.sh"):
            if ".git" not in path.parts:
                result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, f"{path}: {result.stderr}")

    def test_all_python_files_parse(self):
        for path in ROOT.rglob("*.py"):
            if ".git" not in path.parts:
                ast.parse(path.read_text(), filename=str(path))

    def test_variant_keys_do_not_collide(self):
        a = output_utils.sanitize_variant_key("org-a/model")
        b = output_utils.sanitize_variant_key("org-b/model")
        self.assertNotEqual(a, b)
        self.assertNotIn("/", a)

    def test_completion_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            output_utils.write_run_metadata(run, primary_metric=0.5)
            output_utils.complete_run(run)
            data = json.loads((run / "completion.json").read_text())
            self.assertEqual(data["status"], "completed")
            self.assertFalse(list(run.glob("*.tmp")))

    def test_manifest_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp)
            (exp / "experiment.json").write_text((ROOT / "templates/experiment.json").read_text())
            self.assertTrue(preflight.check_manifest(exp))

    def test_resource_estimate_uses_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp = root / "experiments/0001_test"
            exp.mkdir(parents=True)
            (exp / "experiment.json").write_text((ROOT / "templates/experiment.json").read_text())
            log = root / "logs/old/1"
            log.mkdir(parents=True)
            (log / "run_metadata.yaml").write_text("status: COMPLETED\nelapsed_seconds: 7200\nmax_rss: 40G\n")
            result = resource_estimate.estimate(root, exp)
            self.assertEqual(result["recommended"]["memory_gb"], 48)
            self.assertEqual(result["recommended"]["time_hours"], 2.5)

    def test_required_templates_exist(self):
        for name in ("experiment.py", "experiment.json", "config.yml", "review.md", "run_slurm.sh", "run_qsub.sh", "run_local.sh"):
            self.assertTrue((ROOT / "templates" / name).is_file(), name)

    def test_submit_has_all_schedulers(self):
        text = (ROOT / "tools/submit.sh").read_text()
        for scheduler in ("slurm)", "pbs)", "local)"):
            self.assertIn(scheduler, text)

    def test_control_metadata_needs_no_pyyaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_metadata.yaml"
            metadata.save(path, {"status": "RUNNING", "reason": "value: with colon"})
            self.assertEqual(metadata.load(path)["status"], "RUNNING")
            self.assertEqual(metadata.load(path)["reason"], "value: with colon")

    def test_resource_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp)
            for name in ("experiment.json", "run_slurm.sh", "run_qsub.sh"):
                (exp / name).write_text((ROOT / "templates" / name).read_text())
            render_resources.render(exp, "testnode")
            slurm = (exp / "run_slurm.sh").read_text()
            self.assertIn("#SBATCH --partition=x-large-testnode", slurm)
            self.assertIn("#SBATCH --time=24:00:00", slurm)

    def test_no_gnu_only_shell_primitives(self):
        shell = "\n".join(path.read_text() for path in ROOT.rglob("*.sh") if ".git" not in path.parts)
        self.assertNotIn("grep -oP", shell)
        self.assertNotIn("find -printf", shell)
        self.assertNotIn("date --iso-8601", shell)

    def test_research_state_files_and_status(self):
        self.assertTrue((ROOT / "RESEARCH.md").is_file())
        self.assertTrue((ROOT / "ROADMAP.md").is_file())
        status = research_status.render(ROOT)
        self.assertIn("Research question", status)
        self.assertIn("Current objective", status)
        self.assertIn("Decisions needed", status)

    def test_reports_consume_research_state(self):
        text = (ROOT / "tools/report.py").read_text()
        self.assertIn("RESEARCH.md", text)
        self.assertIn("ROADMAP.md", text)

    def test_redundant_documents_are_removed(self):
        for name in ("EXPERIMENT_NOTES.md", "USAGE.md", "FUNCTIONS.md"):
            self.assertFalse((ROOT / name).exists(), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
