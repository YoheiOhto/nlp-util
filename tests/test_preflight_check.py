import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.preflight_check import check_config_experiment_match, check_direct_writes


class ConfigExperimentMatchTests(unittest.TestCase):
    def test_existing_keys_in_index_and_get_calls_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_path = Path(tmp_dir)
            (exp_path / "config.yml").write_text("model: gemma\ndataset: BC5CDR\n")
            (exp_path / "experiment.py").write_text(
                'model = config["model"]\n'
                "dataset = config.get('dataset', 'default')\n"
            )

            self.assertTrue(check_config_experiment_match(exp_path))

    def test_undefined_key_in_get_call_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_path = Path(tmp_dir)
            (exp_path / "config.yml").write_text("model: gemma\n")
            (exp_path / "experiment.py").write_text(
                'model = config["model"]\n'
                'dataset = config.get("dataset")\n'
            )

            output = StringIO()
            with redirect_stdout(output):
                result = check_config_experiment_match(exp_path)

            self.assertFalse(result)
            self.assertIn("dataset", output.getvalue())


class DirectWritesTests(unittest.TestCase):
    def test_commented_write_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_path = Path(tmp_dir)
            (exp_path / "experiment.py").write_text(
                '# df.to_csv("outputs/results.csv")\n'
                'result = 1  # open("data/result.csv", "w")\n'
            )

            self.assertTrue(check_direct_writes(exp_path))

    def test_real_write_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_path = Path(tmp_dir)
            (exp_path / "experiment.py").write_text(
                'df.to_csv("./outputs/results.csv")\n'
            )

            output = StringIO()
            with redirect_stdout(output):
                result = check_direct_writes(exp_path)

            self.assertFalse(result)
            self.assertIn("outputs/results.csv", output.getvalue())

    def test_invalid_python_is_rejected_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_path = Path(tmp_dir)
            (exp_path / "experiment.py").write_text("def broken(:\n")

            output = StringIO()
            with redirect_stdout(output):
                result = check_direct_writes(exp_path)

            self.assertFalse(result)
            self.assertIn("Invalid Python syntax", output.getvalue())


if __name__ == "__main__":
    unittest.main()
