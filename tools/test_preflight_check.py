from tools.preflight_check import check_config_experiment_match


def test_config_reference_forms_are_accepted(tmp_path):
    (tmp_path / "config.yml").write_text("alpha: 1\nbeta: 2\n", encoding="utf-8")
    (tmp_path / "experiment.py").write_text(
        "config.get('alpha')\nconfig[\"beta\"]\n",
        encoding="utf-8",
    )

    assert check_config_experiment_match(tmp_path)


def test_missing_config_reference_is_rejected(tmp_path):
    (tmp_path / "config.yml").write_text("alpha: 1\n", encoding="utf-8")
    (tmp_path / "experiment.py").write_text(
        "config.get('missing')\n",
        encoding="utf-8",
    )

    assert not check_config_experiment_match(tmp_path)
