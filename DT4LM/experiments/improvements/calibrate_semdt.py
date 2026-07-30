#!/usr/bin/env python
"""Run or resume SemDT calibration for exactly one judge backend."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


def _resolve(project_root, value):
    """Interpret pipeline paths relative to the checked-out DT4LM root."""

    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _load_yaml(path, label):
    """Load a mapping-valued YAML file without exposing secret values."""

    with open(path, encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a YAML mapping.")
    return value


def _require(mapping, key, context):
    """Read a required calibration value instead of using a hidden default."""

    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{context} is missing required field {key!r}.")
    return mapping[key]


def _validate_config(config):
    """Validate all hyperparameters consumed by the formal calibration path."""

    for section in ("dataset", "models", "manifests", "nli", "calibration"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Dataset configuration requires section {section!r}.")
    calibration = config["calibration"]
    for key in (
        "output_root",
        "candidate_collection",
        "candidate_sample_size",
        "search_sample_size",
        "minimum_validation_positives",
        "maximum_total_labels",
        "trajectory_sample_size",
        "annotation_batch_size",
        "threshold_search",
    ):
        _require(calibration, key, "calibration")

    collection = calibration["candidate_collection"]
    if not isinstance(collection, dict):
        raise ValueError("calibration.candidate_collection must be a mapping.")
    for key in ("base_recipe", "differential_objective", "semantic_constraint"):
        _require(collection, key, "calibration.candidate_collection")
    if collection["semantic_constraint"] != "original":
        raise ValueError(
            "Calibration candidate collection currently requires the configured "
            "semantic_constraint: original."
        )

    sample_size = calibration["candidate_sample_size"]
    search_size = calibration["search_sample_size"]
    if not isinstance(sample_size, int) or sample_size <= 1:
        raise ValueError("calibration.candidate_sample_size must exceed one.")
    if not isinstance(search_size, int) or not 0 < search_size < sample_size:
        raise ValueError(
            "calibration.search_sample_size must leave a non-empty validation set."
        )
    maximum_labels = calibration["maximum_total_labels"]
    if not isinstance(maximum_labels, int) or maximum_labels < sample_size:
        raise ValueError(
            "calibration.maximum_total_labels cannot be smaller than the "
            "initial candidate sample."
        )

    threshold_search = calibration["threshold_search"]
    if not isinstance(threshold_search, dict):
        raise ValueError("calibration.threshold_search must be a mapping.")
    for key in ("method", "step", "min_precision", "bootstrap_samples"):
        _require(threshold_search, key, "calibration.threshold_search")
    if threshold_search["method"] != "grid":
        raise ValueError(
            "This implementation supports threshold_search.method: grid."
        )

    for key in (
        "model",
        "dtype",
        "batch_size",
        "candidate_batch_size",
        "max_length",
        "truncation_strategy",
    ):
        _require(config["nli"], key, "nli")
    for key in ("seed", "query_budget"):
        _require(config, key, "Dataset configuration")


def _judge_identity(judge_config):
    """Return the selected backend and model without retaining credentials."""

    backend = _require(judge_config, "backend", "Judge configuration")
    backend_config = judge_config.get(backend)
    if not isinstance(backend_config, dict):
        raise ValueError(
            f"Judge configuration requires a {backend!r} mapping for its backend."
        )
    model = _require(backend_config, "model", f"Judge backend {backend}")
    return str(backend), str(model)


def _run(project_root, *arguments):
    """Use the installed local package for every recoverable pipeline stage."""

    subprocess.run(
        [sys.executable, "-m", "textattack", "semdt-calibrate", *arguments],
        cwd=project_root,
        check=True,
    )


def _jsonl_ids(path):
    """Read candidate IDs from a resumable JSONL artifact."""

    if not path.exists():
        return set()
    result = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.add(json.loads(line)["candidate_id"])
    return result


def _verify_annotation_identity(path, backend, model):
    """Prevent a resumable file from mixing judge backends or model versions."""

    if not path.exists():
        return
    identities = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                identities.add((row.get("backend"), row.get("model")))
    if identities and identities != {(backend, model)}:
        raise ValueError(
            f"Existing annotations at {path} use "
            f"{sorted(identities, key=str)!r}; "
            f"the selected judge is {(backend, model)!r}. Use a new output root."
        )


def _annotate_if_needed(
    config,
    project_root,
    judge_config_path,
    input_path,
    output_path,
    backend,
    model,
):
    """Avoid loading an API/HF judge when every input already has a result."""

    _verify_annotation_identity(output_path, backend, model)
    if _jsonl_ids(input_path) <= _jsonl_ids(output_path):
        return
    _run(
        project_root,
        "--stage",
        "annotate",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--judge-config",
        str(judge_config_path),
        "--task-definition",
        str(config["dataset"]["task_definition"]),
        "--dataset",
        str(config["dataset"]["id"]),
        "--annotation-batch-size",
        str(config["calibration"]["annotation_batch_size"]),
    )


def _collect_candidates(config, project_root, raw_path):
    """Collect Base candidates using config-defined objective and constraints."""

    if raw_path.exists():
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = config["dataset"]
    models = config["models"]
    collection = config["calibration"]["candidate_collection"]
    manifest = _resolve(project_root, config["manifests"]["train"])
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Calibration manifest does not exist: {manifest}. "
            "Run prepare_manifests.sh first."
        )
    command = [
        sys.executable,
        "-m",
        "textattack",
        "attack",
        "--recipe",
        "pair",
        "--base-recipe",
        str(collection["base_recipe"]),
        "--differential-objective",
        str(collection["differential_objective"]),
        "--semantic-constraint",
        str(collection["semantic_constraint"]),
        "--num-examples",
        "-1",
        "--query-budget",
        str(config["query_budget"]),
        "--random-seed",
        str(config["seed"]),
        "--model",
        str(_resolve(project_root, models["new"])),
        "--second-model",
        str(_resolve(project_root, models["old"])),
        "--dataset-from-huggingface",
        str(dataset["textattack_spec"]),
        "--dataset-split",
        str(dataset["calibration_split"]),
        "--sample-manifest",
        str(manifest),
        "--candidate-log",
        str(raw_path),
        "--disable-stdout",
    ]
    for option, key in (
        ("--model-revision", "new_revision"),
        ("--second-model-revision", "old_revision"),
    ):
        if models.get(key):
            command.extend([option, str(models[key])])
    subprocess.run(command, cwd=project_root, check=True)


def _verify_frozen_split(split_manifest, calibration, configured_seed):
    """Reject silent reuse when configured split sizes or seed have changed."""

    with open(split_manifest, encoding="utf-8") as handle:
        frozen = json.load(handle)
    expected_search = calibration["search_sample_size"]
    expected_validation = (
        calibration["candidate_sample_size"] - expected_search
    )
    actual = (len(frozen["search_ids"]), len(frozen["validation_ids"]))
    expected = (expected_search, expected_validation)
    if actual != expected:
        raise ValueError(
            f"Existing calibration split has sizes {actual}, expected {expected}. "
            "Use a new calibration.output_root after changing split hyperparameters."
        )
    if int(frozen["seed"]) != int(configured_seed):
        raise ValueError(
            f"Existing calibration split uses seed {frozen['seed']}, "
            f"not configured seed {configured_seed}. Use a new output root."
        )


def _prepare_shared_candidates(config, project_root, calibration_root):
    """Collect, NLI-score, and freeze the configured judge-neutral split."""

    raw_path = calibration_root / "base_candidates.jsonl"
    scored_path = calibration_root / "base_candidates_scored.jsonl"
    split_dir = calibration_root / "split"
    calibration = config["calibration"]
    _collect_candidates(config, project_root, raw_path)

    nli = config["nli"]
    if not scored_path.exists():
        arguments = [
            "--stage",
            "nli-score",
            "--input",
            str(raw_path),
            "--output",
            str(scored_path),
            "--profile-output",
            str(calibration_root / "offline_nli_profile.json"),
            "--candidate-batch-size",
            str(nli["candidate_batch_size"]),
            "--nli-model",
            str(nli["model"]),
            "--nli-dtype",
            str(nli["dtype"]),
            "--nli-batch-size",
            str(nli["batch_size"]),
            "--nli-max-length",
            str(nli["max_length"]),
            "--nli-truncation-strategy",
            str(nli["truncation_strategy"]),
        ]
        if nli.get("revision"):
            arguments.extend(["--nli-model-revision", str(nli["revision"])])
        if nli.get("tokenizer_revision"):
            arguments.extend(
                ["--nli-tokenizer-revision", str(nli["tokenizer_revision"])]
            )
        if nli.get("device"):
            arguments.extend(["--nli-device", str(nli["device"])])
        _run(project_root, *arguments)

    split_manifest = split_dir / "split_manifest.json"
    if not split_manifest.exists():
        _run(
            project_root,
            "--stage",
            "freeze-split",
            "--input",
            str(scored_path),
            "--output-dir",
            str(split_dir),
            "--sample-size",
            str(calibration["candidate_sample_size"]),
            "--search-size",
            str(calibration["search_sample_size"]),
            "--seed",
            str(config["seed"]),
        )
    _verify_frozen_split(split_manifest, calibration, config["seed"])
    return scored_path, split_dir


def _verify_threshold_identity(path, backend, model, threshold_search):
    """Reject a frozen threshold produced by another judge or search setup."""

    if not path.exists():
        return
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    actual = {
        "backend": artifact.get("judge_backend"),
        "model": artifact.get("judge_model"),
        "method": artifact.get("threshold_search_method", "grid"),
        "step": artifact.get("threshold_step", 0.01),
        "min_precision": artifact.get("min_precision"),
    }
    expected = {
        "backend": backend,
        "model": model,
        "method": threshold_search["method"],
        "step": threshold_search["step"],
        "min_precision": threshold_search["min_precision"],
    }
    if actual != expected:
        raise ValueError(
            f"Existing threshold at {path} uses {actual!r}, expected {expected!r}. "
            "Use a new calibration.output_root for changed settings."
        )


def _calibrate_backend(
    config,
    project_root,
    calibration_root,
    split_dir,
    scored_path,
    backend,
    model,
    judge_config_path,
):
    """Annotate and freeze one backend's threshold without cross-label mixing."""

    backend_dir = calibration_root / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    calibration = config["calibration"]
    threshold_search = calibration["threshold_search"]

    for partition in ("search", "validation"):
        _annotate_if_needed(
            config,
            project_root,
            judge_config_path,
            split_dir / f"{partition}.jsonl",
            backend_dir / f"{partition}_labels.jsonl",
            backend,
            model,
        )

    threshold = backend_dir / "threshold.json"
    report = backend_dir / "validation_report.json"
    _verify_threshold_identity(threshold, backend, model, threshold_search)
    if not threshold.exists() or not report.exists():
        _run(
            project_root,
            "--stage",
            "tune-validate",
            "--search-candidates",
            str(split_dir / "search.jsonl"),
            "--validation-candidates",
            str(split_dir / "validation.jsonl"),
            "--search-labels",
            str(backend_dir / "search_labels.jsonl"),
            "--validation-labels",
            str(backend_dir / "validation_labels.jsonl"),
            "--split-manifest",
            str(split_dir / "split_manifest.json"),
            "--threshold-output",
            str(threshold),
            "--report-output",
            str(report),
            "--dataset",
            str(config["dataset"]["id"]),
            "--threshold-search-method",
            str(threshold_search["method"]),
            "--threshold-step",
            str(threshold_search["step"]),
            "--min-precision",
            str(threshold_search["min_precision"]),
            "--bootstrap-samples",
            str(threshold_search["bootstrap_samples"]),
            "--minimum-validation-positives",
            str(calibration["minimum_validation_positives"]),
            "--seed",
            str(config["seed"]),
        )

    supplemental = backend_dir / "supplemental_audit.jsonl"
    _run(
        project_root,
        "--stage",
        "supplemental-audit",
        "--input",
        str(scored_path),
        "--sampled-candidates",
        str(split_dir / "sampled.jsonl"),
        "--validation-labels",
        str(backend_dir / "validation_labels.jsonl"),
        "--output",
        str(supplemental),
        "--minimum-validation-positives",
        str(calibration["minimum_validation_positives"]),
        "--maximum-total-labels",
        str(calibration["maximum_total_labels"]),
        "--seed",
        str(config["seed"]),
    )
    if supplemental.exists() and supplemental.stat().st_size:
        supplemental_labels = backend_dir / "supplemental_labels.jsonl"
        _annotate_if_needed(
            config,
            project_root,
            judge_config_path,
            supplemental,
            supplemental_labels,
            backend,
            model,
        )
        supplemental_report = backend_dir / "supplemental_report.json"
        if not supplemental_report.exists():
            _run(
                project_root,
                "--stage",
                "frozen-report",
                "--input",
                str(supplemental),
                "--validation-labels",
                str(supplemental_labels),
                "--threshold-output",
                str(threshold),
                "--output",
                str(supplemental_report),
                "--bootstrap-samples",
                str(threshold_search["bootstrap_samples"]),
                "--seed",
                str(config["seed"]),
            )


def _audit_trajectory(
    config,
    project_root,
    calibration_root,
    backend,
    model,
    judge_config_path,
    run_dir,
):
    """Audit one explicitly selected formal run without retuning thresholds."""

    trajectory = run_dir / "nli_candidates.jsonl"
    if not trajectory.exists():
        raise FileNotFoundError(
            f"Selected trajectory run has no NLI audit stream: {trajectory}."
        )
    calibration = config["calibration"]
    threshold_search = calibration["threshold_search"]
    backend_dir = calibration_root / backend
    audit_dir = backend_dir / "trajectory_audits" / run_dir.name
    audit_dir.mkdir(parents=True, exist_ok=True)
    sampled = audit_dir / "candidates.jsonl"
    labels = audit_dir / "labels.jsonl"
    report = audit_dir / "report.json"

    if not sampled.exists():
        _run(
            project_root,
            "--stage",
            "trajectory-sample",
            "--input",
            str(trajectory),
            "--output",
            str(sampled),
            "--trajectory-sample-size",
            str(calibration["trajectory_sample_size"]),
            "--seed",
            str(config["seed"]),
        )
    _annotate_if_needed(
        config,
        project_root,
        judge_config_path,
        sampled,
        labels,
        backend,
        model,
    )
    if not report.exists():
        _run(
            project_root,
            "--stage",
            "trajectory-report",
            "--input",
            str(sampled),
            "--validation-labels",
            str(labels),
            "--threshold-output",
            str(backend_dir / "threshold.json"),
            "--base-candidates",
            str(calibration_root / "split" / "validation.jsonl"),
            "--base-labels",
            str(backend_dir / "validation_labels.jsonl"),
            "--output",
            str(report),
            "--bootstrap-samples",
            str(threshold_search["bootstrap_samples"]),
            "--seed",
            str(config["seed"]),
        )


def main():
    """Execute one backend so API and local-HF failures remain independent."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--judge-config", required=True)
    parser.add_argument(
        "--trajectory-run-dir",
        help="Optional completed SemDT run whose frozen-threshold trajectory is audited.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config = _load_yaml(Path(args.dataset_config).resolve(), "Dataset configuration")
    _validate_config(config)
    judge_config_path = _resolve(project_root, args.judge_config)
    judge_config = _load_yaml(judge_config_path, "Judge configuration")
    backend, model = _judge_identity(judge_config)

    calibration_root = _resolve(
        project_root, config["calibration"]["output_root"]
    )
    calibration_root.mkdir(parents=True, exist_ok=True)
    backend_dir = calibration_root / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    # Persist only non-secret identity fields for reproducibility.
    with open(
        backend_dir / "judge_identity.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {"backend": backend, "model": model},
            handle,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )

    scored_path, split_dir = _prepare_shared_candidates(
        config, project_root, calibration_root
    )
    _calibrate_backend(
        config,
        project_root,
        calibration_root,
        split_dir,
        scored_path,
        backend,
        model,
        judge_config_path,
    )
    if args.trajectory_run_dir:
        _audit_trajectory(
            config,
            project_root,
            calibration_root,
            backend,
            model,
            judge_config_path,
            _resolve(project_root, args.trajectory_run_dir),
        )


if __name__ == "__main__":
    main()
