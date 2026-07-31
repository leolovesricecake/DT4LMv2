#!/usr/bin/env python
"""Run exactly one config-defined DT4LM improvement experiment."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


# Direct script execution exposes only experiments/improvements on sys.path.
# Add the DT4LM root so every pipeline entry point shares artifact rules.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dt4lm_artifacts import (  # noqa: E402
    resolve_model_id,
    resolve_path,
    run_directory,
    validate_artifact_namespaces,
    validate_manifest_identity,
)


OBJECTIVES = {"dynamic", "static", "lexi"}
SEMANTIC_CONSTRAINTS = {"original", "nli"}
THRESHOLD_SOURCES = {"none", "manual", "calibrated"}


def _resolve(project_root, value):
    """Interpret experiment paths relative to the checked-out DT4LM root."""

    return resolve_path(project_root, value)


def _load_yaml(path, label):
    """Load one mapping-valued YAML file with an actionable error."""

    with open(path, encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a YAML mapping.")
    return value


def _require(mapping, key, context):
    """Read a required configuration value instead of hiding a code default."""

    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{context} is missing required field {key!r}.")
    return mapping[key]


def _validate_experiment(experiment):
    """Validate the independent objective, constraint, and threshold axes."""

    for key in (
        "name",
        "base_recipe",
        "differential_objective",
        "semantic_constraint",
        "semantic_threshold",
    ):
        _require(experiment, key, "Experiment configuration")

    objective = experiment["differential_objective"]
    semantic = experiment["semantic_constraint"]
    threshold = experiment["semantic_threshold"]
    if objective not in OBJECTIVES:
        raise ValueError(
            f"Unsupported differential_objective {objective!r}; "
            f"choose from {sorted(OBJECTIVES)}."
        )
    if semantic not in SEMANTIC_CONSTRAINTS:
        raise ValueError(
            f"Unsupported semantic_constraint {semantic!r}; "
            f"choose from {sorted(SEMANTIC_CONSTRAINTS)}."
        )
    if not isinstance(threshold, dict):
        raise ValueError("semantic_threshold must be a YAML mapping.")
    source = _require(threshold, "source", "semantic_threshold")
    if source not in THRESHOLD_SOURCES:
        raise ValueError(
            f"Unsupported semantic threshold source {source!r}; "
            f"choose from {sorted(THRESHOLD_SOURCES)}."
        )
    if semantic == "original" and source != "none":
        raise ValueError("The original semantic constraint requires source: none.")
    if semantic == "nli" and source == "none":
        raise ValueError("The NLI semantic constraint requires a threshold source.")
    if source == "manual":
        _require(threshold, "entailment", "Manual semantic threshold")
        _require(threshold, "contradiction", "Manual semantic threshold")
    if source == "calibrated":
        _require(threshold, "backend", "Calibrated semantic threshold")


def _validate_dataset_config(config, project_root=PROJECT_ROOT):
    """Check fields consumed by every formal experiment invocation."""

    for key in ("seed", "query_budget", "output_root"):
        _require(config, key, "Dataset configuration")
    for section in ("dataset", "models", "manifests"):
        if not isinstance(config.get(section), dict):
            raise ValueError(
                f"Dataset configuration requires mapping section {section!r}."
            )
    for key in ("id", "textattack_spec", "test_split"):
        _require(config["dataset"], key, "dataset")
    for key in ("id", "new", "old"):
        _require(config["models"], key, "models")
    _require(config["manifests"], "test", "manifests")
    validate_artifact_namespaces(config, project_root)


def _environment(project_root):
    """Capture revisions and hardware without querying any external service."""

    def git(*arguments):
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip()

    packages = {}
    for package in ("torch", "transformers", "datasets", "openai"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    hardware = {"cuda_available": False, "cuda_version": None, "gpus": []}
    try:
        import torch

        hardware = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpus": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        pass
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version,
        "packages": packages,
        **hardware,
    }


def _append_option(command, option, value):
    """Append a CLI option only when its configured value is present."""

    if value is not None:
        command.extend([option, str(value)])


def _threshold_file(config, experiment, project_root):
    """Resolve a frozen threshold artifact selected by one experiment YAML."""

    threshold = experiment["semantic_threshold"]
    configured_path = threshold.get("file")
    if configured_path:
        path = _resolve(project_root, configured_path)
    else:
        calibration = config.get("calibration")
        if not isinstance(calibration, dict):
            raise ValueError(
                "A calibrated threshold requires dataset calibration.output_root."
            )
        calibration_root = _resolve(
            project_root,
            _require(calibration, "output_root", "calibration"),
        )
        path = calibration_root / str(threshold["backend"]) / "threshold.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibrated threshold does not exist: {path}. "
            "Run calibrate_semdt.sh for this dataset and backend first."
        )
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    configured_backend = str(threshold["backend"])
    if artifact.get("judge_backend") != configured_backend:
        raise ValueError(
            f"Threshold {path} belongs to backend "
            f"{artifact.get('judge_backend')!r}, not {configured_backend!r}."
        )
    expected_scope = {
        "dataset": str(config["dataset"]["id"]),
        "model_pair_id": str(config["models"]["id"]),
    }
    actual_scope = {key: artifact.get(key) for key in expected_scope}
    if actual_scope != expected_scope:
        raise ValueError(
            f"Threshold {path} belongs to {actual_scope!r}, "
            f"not {expected_scope!r}."
        )
    return path


def _attack_command(config, experiment, run_dir, manifest, project_root):
    """Build one attack command entirely from dataset and experiment YAML."""

    dataset = config["dataset"]
    models = config["models"]
    command = [
        sys.executable,
        "-m",
        "textattack",
        "attack",
        "--recipe",
        "pair",
        "--base-recipe",
        str(experiment["base_recipe"]),
        "--differential-objective",
        str(experiment["differential_objective"]),
        "--semantic-constraint",
        str(experiment["semantic_constraint"]),
        "--num-examples",
        "-1",
        "--query-budget",
        str(config["query_budget"]),
        "--random-seed",
        str(config["seed"]),
        "--model",
        resolve_model_id(project_root, models["new"]),
        "--second-model",
        resolve_model_id(project_root, models["old"]),
        "--dataset-from-huggingface",
        str(dataset["textattack_spec"]),
        "--dataset-split",
        str(dataset["test_split"]),
        "--sample-manifest",
        str(manifest),
        "--log-to-jsonl",
        str(run_dir / "results.jsonl"),
        "--log-summary-to-json",
        str(run_dir / "attack_summary.json"),
        "--disable-stdout",
    ]
    _append_option(command, "--model-revision", models.get("new_revision"))
    _append_option(command, "--second-model-revision", models.get("old_revision"))

    if experiment["semantic_constraint"] == "nli":
        nli = config.get("nli")
        if not isinstance(nli, dict):
            raise ValueError("NLI experiments require an nli mapping in dataset YAML.")
        for key in (
            "model",
            "dtype",
            "batch_size",
            "max_length",
            "truncation_strategy",
        ):
            _require(nli, key, "nli")
        command.extend(
            [
                "--nli-model-name-or-path",
                str(nli["model"]),
                "--nli-dtype",
                str(nli["dtype"]),
                "--nli-batch-size",
                str(nli["batch_size"]),
                "--nli-max-length",
                str(nli["max_length"]),
                "--nli-truncation-strategy",
                str(nli["truncation_strategy"]),
                "--nli-audit-log",
                str(run_dir / "nli_candidates.jsonl"),
                "--nli-profile-output",
                str(run_dir / "nli_profile.json"),
            ]
        )
        _append_option(command, "--nli-model-revision", nli.get("revision"))
        _append_option(
            command, "--nli-tokenizer-revision", nli.get("tokenizer_revision")
        )
        _append_option(command, "--nli-device", nli.get("device"))

        threshold = experiment["semantic_threshold"]
        if threshold["source"] == "calibrated":
            command.extend(
                [
                    "--semantic-threshold-file",
                    str(_threshold_file(config, experiment, project_root)),
                ]
            )
        elif threshold["source"] == "manual":
            command.extend(
                [
                    "--nli-entailment-threshold",
                    str(threshold["entailment"]),
                    "--nli-contradiction-threshold",
                    str(threshold["contradiction"]),
                ]
            )
    return command


def _split_results(run_dir):
    """Store individual records without changing the canonical JSONL stream."""

    success_dir = run_dir / "successful_examples"
    failed_dir = run_dir / "failed_examples"
    success_dir.mkdir(exist_ok=True)
    failed_dir.mkdir(exist_ok=True)
    with open(run_dir / "results.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            target = success_dir if row["success"] else failed_dir
            with open(
                target / f"{row['dataset_index']}.json",
                "w",
                encoding="utf-8",
            ) as output:
                json.dump(row, output, ensure_ascii=True, indent=2, sort_keys=True)


def main():
    """Parse two configs and execute one independently recoverable experiment."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--skip-bertscore", action="store_true")
    args = parser.parse_args()

    dataset_config_path = Path(args.dataset_config).resolve()
    experiment_config_path = Path(args.experiment_config).resolve()
    config = _load_yaml(dataset_config_path, "Dataset configuration")
    experiment = _load_yaml(experiment_config_path, "Experiment configuration")
    _validate_dataset_config(config)
    _validate_experiment(experiment)

    project_root = PROJECT_ROOT
    manifest = _resolve(project_root, config["manifests"]["test"])
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Test manifest does not exist: {manifest}. "
            "Run prepare_manifests.sh first."
        )
    validate_manifest_identity(
        manifest,
        config,
        config["dataset"]["test_split"],
        project_root,
    )

    # Dataset and model-pair namespaces prevent independent experiment families
    # from colliding even when they reuse the same experiment name.
    run_dir = run_directory(config, project_root, experiment["name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "results.jsonl").exists():
        raise FileExistsError(
            f"{run_dir} already has results; change experiment.name or output_root."
        )
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "dataset_config": config,
                "dataset_config_path": str(dataset_config_path),
                "experiment_config": experiment,
                "experiment_config_path": str(experiment_config_path),
            },
            handle,
            allow_unicode=False,
            sort_keys=True,
        )
    with open(run_dir / "environment.json", "w", encoding="utf-8") as handle:
        json.dump(
            _environment(project_root),
            handle,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    shutil.copy2(manifest, run_dir / "sample_manifest.json")

    subprocess.run(
        _attack_command(config, experiment, run_dir, manifest, project_root),
        cwd=project_root,
        check=True,
    )
    _split_results(run_dir)

    evaluator = [
        sys.executable,
        str(project_root / "statistics" / "evaluate_improvements.py"),
        "--results",
        str(run_dir / "results.jsonl"),
        "--manifest",
        str(run_dir / "sample_manifest.json"),
        "--output",
        str(run_dir / "summary.json"),
        "--method",
        str(experiment["name"]),
    ]
    if (run_dir / "nli_profile.json").exists():
        evaluator.extend(["--nli-profile", str(run_dir / "nli_profile.json")])
    baseline_name = experiment.get("comparison_baseline")
    if baseline_name:
        base_summary = run_dir.parent / str(baseline_name) / "summary.json"
        if base_summary.exists():
            evaluator.extend(["--base-summary", str(base_summary)])
    if args.skip_bertscore:
        evaluator.append("--skip-bertscore")
    subprocess.run(evaluator, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
