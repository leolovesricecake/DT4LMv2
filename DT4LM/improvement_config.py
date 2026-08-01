"""Validation helpers for self-contained DT4LM improvement experiments."""

from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = 2
OBJECTIVES = frozenset(("dynamic", "static", "lexi"))
SEMANTIC_CONSTRAINTS = frozenset(("original", "nli"))
THRESHOLD_SOURCES = frozenset(("none", "manual", "calibrated"))


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    """Return one required value with a configuration-local error message."""

    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{context} is missing required field {key!r}.")
    return mapping[key]


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Require a top-level mapping before validating its fields."""

    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Experiment configuration requires mapping {key!r}.")
    return value


def model_spec(config: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    """Return the nested old/new model specification."""

    models = _require_mapping(config, "models")
    value = models.get(role)
    if not isinstance(value, dict):
        raise ValueError(f"models.{role} must be a mapping.")
    _require(value, "name_or_path", f"models.{role}")
    return value


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    """Validate every field needed before an experiment starts model queries."""

    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}; got "
            f"{config.get('schema_version')!r}."
        )

    experiment = _require_mapping(config, "experiment")
    for key in ("id", "method", "seed", "output_root"):
        _require(experiment, key, "experiment")
    if not isinstance(experiment["seed"], int):
        raise ValueError("experiment.seed must be an integer.")

    dataset = _require_mapping(config, "dataset")
    for key in (
        "id",
        "path",
        "text_columns",
        "label_column",
        "task_definition",
    ):
        _require(dataset, key, "dataset")
    text_columns = dataset["text_columns"]
    if (
        not isinstance(text_columns, list)
        or not text_columns
        or any(not isinstance(column, str) or not column for column in text_columns)
        or len(set(text_columns)) != len(text_columns)
    ):
        raise ValueError(
            "dataset.text_columns must be a non-empty list of unique strings."
        )
    label_column = dataset["label_column"]
    if not isinstance(label_column, str) or not label_column:
        raise ValueError("dataset.label_column must be a non-empty string.")
    if label_column in text_columns:
        raise ValueError("dataset.label_column cannot also be a text column.")
    if (
        not isinstance(dataset["task_definition"], str)
        or not dataset["task_definition"].strip()
    ):
        raise ValueError("dataset.task_definition must be a non-empty string.")
    evaluation_sample = dataset.get("evaluation")
    if not isinstance(evaluation_sample, dict):
        raise ValueError("dataset.evaluation must be a mapping.")
    for key in ("split", "sample_seed", "manifest"):
        _require(evaluation_sample, key, "dataset.evaluation")
    sample_size = evaluation_sample.get("sample_size")
    if sample_size is not None and not isinstance(sample_size, int):
        raise ValueError("dataset.evaluation.sample_size must be an integer or null.")

    models = _require_mapping(config, "models")
    _require(models, "id", "models")
    model_spec(config, "old")
    model_spec(config, "new")

    attack = _require_mapping(config, "attack")
    for key in (
        "recipe",
        "differential_objective",
        "semantic_constraint",
        "query_budget",
    ):
        _require(attack, key, "attack")
    if attack["differential_objective"] not in OBJECTIVES:
        raise ValueError(
            "attack.differential_objective must be one of "
            f"{sorted(OBJECTIVES)!r}."
        )
    if attack["semantic_constraint"] not in SEMANTIC_CONSTRAINTS:
        raise ValueError(
            "attack.semantic_constraint must be one of "
            f"{sorted(SEMANTIC_CONSTRAINTS)!r}."
        )
    if not isinstance(attack["query_budget"], int) or attack["query_budget"] <= 0:
        raise ValueError("attack.query_budget must be a positive integer.")

    semantic = _require_mapping(config, "semantic")
    threshold = semantic.get("threshold")
    if not isinstance(threshold, dict):
        raise ValueError("semantic.threshold must be a mapping.")
    source = _require(threshold, "source", "semantic.threshold")
    if source not in THRESHOLD_SOURCES:
        raise ValueError(
            f"semantic.threshold.source must be one of {sorted(THRESHOLD_SOURCES)!r}."
        )
    if attack["semantic_constraint"] == "original" and source != "none":
        raise ValueError("The original semantic constraint requires source: none.")
    if attack["semantic_constraint"] == "nli":
        if source == "none":
            raise ValueError("The NLI semantic constraint requires a threshold.")
        nli = semantic.get("nli")
        if not isinstance(nli, dict):
            raise ValueError("NLI experiments require semantic.nli.")
        for key in (
            "model_name_or_path",
            "dtype",
            "batch_size",
            "max_length",
            "truncation_strategy",
        ):
            _require(nli, key, "semantic.nli")
    if source == "manual":
        _require(threshold, "entailment", "semantic.threshold")
        _require(threshold, "contradiction", "semantic.threshold")
    if source == "calibrated":
        _require(threshold, "backend", "semantic.threshold")
        _require(threshold, "artifact", "semantic.threshold")

    calibration = _require_mapping(config, "calibration")
    enabled = _require(calibration, "enabled", "calibration")
    if not isinstance(enabled, bool):
        raise ValueError("calibration.enabled must be true or false.")
    if enabled:
        for key in (
            "output_dir",
            "source_sampling",
            "candidate_collection",
            "candidate_sample_size",
            "search_sample_size",
            "minimum_validation_positives",
            "maximum_total_labels",
            "trajectory_sample_size",
            "annotation_batch_size",
            "threshold_search",
            "judge",
        ):
            _require(calibration, key, "calibration")
        source_sampling = calibration["source_sampling"]
        if not isinstance(source_sampling, dict):
            raise ValueError("calibration.source_sampling must be a mapping.")
        for key in ("split", "sample_seed", "manifest"):
            _require(source_sampling, key, "calibration.source_sampling")
        source_size = source_sampling.get("sample_size")
        if source_size is not None and not isinstance(source_size, int):
            raise ValueError(
                "calibration.source_sampling.sample_size must be an integer or null."
            )
        candidate_size = calibration["candidate_sample_size"]
        search_size = calibration["search_sample_size"]
        if not isinstance(candidate_size, int) or candidate_size <= 1:
            raise ValueError("calibration.candidate_sample_size must exceed one.")
        if not isinstance(search_size, int) or not 0 < search_size < candidate_size:
            raise ValueError(
                "calibration.search_sample_size must leave a validation partition."
            )
        judge = calibration["judge"]
        if not isinstance(judge, dict):
            raise ValueError("calibration.judge must be a mapping.")
        for key in ("backend", "config_file"):
            _require(judge, key, "calibration.judge")
        search = calibration["threshold_search"]
        if not isinstance(search, dict):
            raise ValueError("calibration.threshold_search must be a mapping.")
        for key in ("method", "step", "min_precision", "bootstrap_samples"):
            _require(search, key, "calibration.threshold_search")
        if search["method"] != "grid":
            raise ValueError("Only calibration.threshold_search.method: grid is supported.")

    evaluation = _require_mapping(config, "evaluation")
    core = evaluation.get("core")
    quality = evaluation.get("quality")
    if not isinstance(core, dict) or not isinstance(quality, dict):
        raise ValueError("evaluation requires core and quality mappings.")
    budgets = _require(core, "success_budgets", "evaluation.core")
    if not isinstance(budgets, list) or not budgets or any(
        not isinstance(value, int) or value <= 0 for value in budgets
    ):
        raise ValueError("evaluation.core.success_budgets must contain positive integers.")
    for metric in ("bleu", "meteor", "rouge_l", "bertscore"):
        metric_config = quality.get(metric)
        if not isinstance(metric_config, dict) or not isinstance(
            metric_config.get("enabled"), bool
        ):
            raise ValueError(f"evaluation.quality.{metric}.enabled must be boolean.")
    bertscore = quality["bertscore"]
    if bertscore["enabled"]:
        for key in (
            "model_name_or_path",
            "num_layers",
            "allow_remote_download",
            "batch_size",
            "idf",
            "rescale_with_baseline",
        ):
            _require(bertscore, key, "evaluation.quality.bertscore")


def load_experiment_config(path: Path) -> dict:
    """Load and validate one complete experiment YAML file."""

    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must contain a YAML mapping.")
    validate_experiment_config(config)
    return config
