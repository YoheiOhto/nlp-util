import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CreateExperimentTests(unittest.TestCase):
    def test_micgm1_fallback_selects_existing_time_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project = Path(tmp_dir)
            (project / "tools").mkdir()
            (project / "templates").mkdir()
            (project / "tools/create_exp.sh").write_bytes(
                (ROOT / "tools/create_exp.sh").read_bytes()
            )
            for name in ("run_slurm.sh", "experiment.py", "config.yml"):
                (project / "templates" / name).write_bytes(
                    (ROOT / "templates" / name).read_bytes()
                )
            subprocess.run(["git", "init", "-q", str(project)], check=True)

            fake_bin = project / "fake-bin"
            fake_bin.mkdir()
            hostname = fake_bin / "hostname"
            hostname.write_text("#!/bin/sh\nprintf 'micgm1\\n'\n")
            hostname.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"

            result = subprocess.run(
                ["bash", "tools/create_exp.sh", "partition-check"],
                cwd=project, env=env, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            generated = next((project / "experiments").glob("0001_*/run_slurm.sh"))
            content = generated.read_text()
            self.assertIn("#SBATCH --partition=x-large-micgm1\n", content)
            self.assertIn("#SBATCH --time=196:00:00\n", content)


if __name__ == "__main__":
    unittest.main()
