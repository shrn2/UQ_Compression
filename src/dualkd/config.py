"""Strict configuration loading for the supported experiment grid."""

from __future__ import annotations

from pathlib import Path


SUPPORTED_METHODS = ("kl_only", "dual_kl")
SUPPORTED_DATASETS = ("arc_challenge", "mmlu_aux", "mmlu_pro", "hellaswag")
REQUIRED_PARTITIONS = ("gate_training", "lambda_selection", "evaluation")


def load_config(path: str | Path) -> dict:
    """Load and validate the single supported experiment schema."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise RuntimeError("configuration loading requires PyYAML") from exc

    source = Path(path)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("experiment configuration must be a mapping")
    methods = tuple(config.get("methods", ()))
    datasets = config.get("datasets", {})
    models = config.get("models", {})
    retentions = tuple(float(value) for value in config.get("retentions", ()))
    training = config.get("training", {})
    if methods != SUPPORTED_METHODS:
        raise ValueError(f"methods must be exactly {SUPPORTED_METHODS}")
    if tuple(datasets) != SUPPORTED_DATASETS:
        raise ValueError(f"datasets must be exactly {SUPPORTED_DATASETS}")
    if not models or any(not isinstance(value, str) or not value for value in models.values()):
        raise ValueError("models must map non-empty keys to Hugging Face model names")
    if not retentions or any(not 0.0 < value < 1.0 for value in retentions):
        raise ValueError("retentions must lie strictly between zero and one")
    if len(set(retentions)) != len(retentions):
        raise ValueError("retentions must be unique")
    for key, dataset in datasets.items():
        if not isinstance(dataset, dict):
            raise ValueError(f"dataset {key} must be a mapping")
        expected = dataset.get("expected", {})
        if any(int(expected.get(partition, 0)) < 1 for partition in REQUIRED_PARTITIONS):
            raise ValueError(f"dataset {key} needs positive expected counts for every partition")
    if int(training.get("epochs", 0)) < 1 or int(training.get("per_stratum", 0)) < 1:
        raise ValueError("training epochs and per_stratum must be positive")
    if float(training.get("dual_learning_rate", 0.0)) <= 0.0:
        raise ValueError("dual_learning_rate must be positive")
    config["config_path"] = str(source.resolve())
    config["retentions"] = retentions
    return config


def retention_tag(value: float) -> str:
    return f"retain_{float(value):.2f}".replace(".", "")
