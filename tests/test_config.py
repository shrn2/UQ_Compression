from pathlib import Path

import pytest
import yaml

from dualkd.config import SUPPORTED_DATASETS, SUPPORTED_METHODS, load_config


CONFIG = Path(__file__).resolve().parents[1] / "configs/experiment.yaml"


def test_repository_config_has_exact_supported_scope():
    config = load_config(CONFIG)
    assert tuple(config["methods"]) == SUPPORTED_METHODS
    assert tuple(config["datasets"]) == SUPPORTED_DATASETS
    assert set(config["retentions"]) == {0.25, 0.50, 0.75}


def test_config_rejects_a_third_method(tmp_path):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["methods"].append("ubkl")
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="methods must be exactly"):
        load_config(path)
