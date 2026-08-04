#!/usr/bin/env python
"""Run one self-contained DT4LM improvement experiment."""

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


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
from dt4lm_dataset import (  # noqa: E402
    load_dataset_collection,
    validate_dataset_split_schema,
)
from improvement_config import load_experiment_config  # noqa: E402


def _environment(project_root):
    """Capture revisions and hardware without querying an external service."""

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
    for package in ("torch", "transformers", "datasets", "openai", "bert-score"):
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


def _write_json_atomic(path, payload):
    """Write stage state without exposing partially serialized JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _set_stage(status_path, stage, state, error=None, **details):
    """Update one independently recoverable pipeline stage."""

    if status_path.exists():
        with open(status_path, encoding="utf-8") as handle:
            status = json.load(handle)
    else:
        status = {
            "schema_version": 2,
            "attack": {"status": "pending"},
            "core_evaluation": {"status": "pending"},
            "quality_evaluation": {"status": "pending"},
        }
    entry = {"status": state, "updated_at": datetime.now(timezone.utc).isoformat()}
    if error:
        entry["error"] = str(error)
    entry.update(details)
    status[stage] = entry
    _write_json_atomic(status_path, status)


def _read_json(path):
    """Read one JSON artifact used by resume validation."""

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _attack_identity(config):
    """Select config fields that must remain immutable after attack generation."""

    experiment = config["experiment"]
    return {
        "schema_version": config["schema_version"],
        "experiment": {
            key: experiment[key] for key in ("id", "method", "seed")
        },
        "dataset": config["dataset"],
        "models": config["models"],
        "attack": config["attack"],
        "semantic": config["semantic"],
    }


def _prepare_run_metadata(run_dir, config, manifest_path, manifest_payload):
    """Create metadata once and reject incompatible attempts to reuse a run."""

    resolved_config_path = run_dir / "config.resolved.yaml"
    if resolved_config_path.exists():
        with open(resolved_config_path, encoding="utf-8") as handle:
            previous = yaml.safe_load(handle)
        if _attack_identity(previous) != _attack_identity(config):
            raise ValueError(
                "Existing results use a different attack configuration. Change "
                "experiment.id instead of resuming this run directory."
            )

    copied_manifest = run_dir / "sample_manifest.json"
    if copied_manifest.exists() and _read_json(copied_manifest) != manifest_payload:
        raise ValueError(
            "Existing sample_manifest.json differs from the configured manifest; "
            "the attack cannot be resumed safely."
        )

    # Evaluation settings may be corrected after attack completion, so refresh
    # the resolved config while preserving all attack-defining fields above.
    with open(resolved_config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=False, sort_keys=True)
    if not copied_manifest.exists():
        shutil.copy2(manifest_path, copied_manifest)
    provenance_path = run_dir / "provenance.json"
    if not provenance_path.exists():
        _write_json_atomic(provenance_path, _environment(PROJECT_ROOT))
    return resolved_config_path


def _load_completed_results(run_dir, manifest):
    """Require a complete ordered result artifact before skipping attack work."""

    records = []
    with open(run_dir / "results.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    expected = [int(index) for index in manifest["selected_indices"]]
    observed = [int(row.get("dataset_index", -1)) for row in records]
    if len(records) != int(manifest["effective_sample_size"]) or observed != expected:
        raise ValueError(
            "Existing results.jsonl is incomplete or does not match manifest order; "
            "use a new experiment.id rather than overwriting partial results."
        )
    if any(row.get("schema_version") != 4 for row in records):
        raise ValueError(
            "Existing results.jsonl predates the FF-PBS schema-v4 protocol; "
            "remove or archive the old run directory before rerunning."
        )
    invalid = [
        row.get("result_status")
        for row in records
        if row.get("result_status") not in {"successful", "failed", "skipped"}
    ]
    if invalid:
        raise ValueError(f"Existing results contain invalid statuses: {invalid!r}.")
    return records


def _load_schema_artifact(path):
    """Return a metric artifact only when it is valid schema-v4 JSON."""

    try:
        payload = _read_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if payload.get("schema_version") == 4 else None


def _core_evaluation_complete(run_dir, core_config, query_budget):
    """Check both scalar and curve-data artifacts before skipping core metrics."""

    if (run_dir / "metrics" / "resources.json").exists():
        return False
    core = _load_schema_artifact(run_dir / "metrics" / "core.json")
    queries = _load_schema_artifact(run_dir / "metrics" / "query_data.json")
    if not core or not queries or not isinstance(core.get("resources"), dict):
        return False
    try:
        actual_budgets = {
            int(key.removeprefix("success_at_"))
            for key in core
            if key.startswith("success_at_")
        }
    except ValueError:
        return False
    expected_budgets = set(core_config["success_budgets"])
    if core.get("query_budget") != query_budget or actual_budgets != expected_budgets:
        return False
    data = queries.get("data") or {}
    columns = (
        data.get("dataset_index"),
        data.get("result_status"),
        data.get("model_pair_queries"),
        data.get("queries_to_success"),
        data.get("budget_penalized_queries"),
    )
    expected = core.get("total")
    return all(isinstance(column, list) for column in columns) and all(
        len(column) == expected for column in columns
    )


def _quality_evaluation_complete(run_dir, quality_config):
    """Retry quality evaluation until every enabled metric has completed."""

    if any(
        (run_dir / "metrics" / f"{name}.json").exists()
        for name in ("bleu", "meteor", "rouge_l", "bertscore")
    ):
        return False
    quality = _load_schema_artifact(run_dir / "metrics" / "quality.json")
    core = _load_schema_artifact(run_dir / "metrics" / "core.json")
    if (
        not quality
        or not core
        or quality.get("status") != "completed"
        or quality.get("successful_sample_count") != core.get("successful")
    ):
        return False
    metrics = quality.get("metrics") or {}
    for name, config in quality_config.items():
        expected = "completed" if config["enabled"] else "disabled"
        payload = metrics.get(name) or {}
        if payload.get("status") != expected or payload.get("config") != config:
            return False
    return True


def _append_option(command, option, value):
    """Append a CLI option only when its configured value is present."""

    if value is not None:
        command.extend([option, str(value)])


def _threshold_file(config, project_root):
    """Resolve and validate a frozen threshold selected by this experiment."""

    threshold = config["semantic"]["threshold"]
    path = resolve_path(project_root, threshold["artifact"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibrated threshold does not exist: {path}. "
            "Run calibrate_semdt.sh with this experiment config first."
        )
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    expected = {
        "judge_backend": str(threshold["backend"]),
        "dataset": str(config["dataset"]["id"]),
        "model_pair_id": str(config["models"]["id"]),
    }
    mismatches = {
        key: {"artifact": artifact.get(key), "configured": value}
        for key, value in expected.items()
        if artifact.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Threshold identity does not match config: {mismatches!r}.")
    return path


def _dataset_argument(config, project_root):
    """Resolve local datasets while preserving Hub dataset identifiers."""

    configured = str(config["dataset"]["path"])
    candidate = Path(configured).expanduser()
    local = candidate if candidate.is_absolute() else project_root / candidate
    return str(local.resolve()) if local.exists() else configured


def _recipe_parameters(config, project_root):
    """Resolve auxiliary model paths before serializing recipe parameters."""

    recipe = str(config["attack"]["recipe"])
    parameters = copy.deepcopy(config["attack"]["recipe_parameters"])
    if recipe == "kuleshov_var":
        parameters["fluency_model_name_or_path"] = resolve_model_id(
            project_root, parameters["fluency_model_name_or_path"]
        )
    if recipe == "faster-alzantot" and parameters["language_model_path"]:
        model_path = resolve_path(project_root, parameters["language_model_path"])
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"FastGA Learning-to-Write model directory does not exist: "
                f"{model_path}."
            )
        parameters["language_model_path"] = str(model_path)
    return parameters


def _attack_command(config, run_dir, manifest, project_root):
    """Build one attack command from the complete experiment YAML."""

    dataset = config["dataset"]
    models = config["models"]
    attack = config["attack"]
    experiment = config["experiment"]
    command = [
        sys.executable,
        "-m",
        "textattack",
        "attack",
        "--recipe",
        "pair",
        "--base-recipe",
        str(attack["recipe"]),
        "--base-recipe-parameters",
        json.dumps(_recipe_parameters(config, project_root), sort_keys=True),
        "--differential-objective",
        str(attack["differential_objective"]),
        "--semantic-constraint",
        str(attack["semantic_constraint"]),
        "--num-examples",
        "-1",
        "--query-budget",
        str(attack["query_budget"]),
        "--model-batch-size",
        str(attack["model_batch_size"]),
        "--random-seed",
        str(experiment["seed"]),
        "--model",
        resolve_model_id(project_root, models["new"]),
        "--second-model",
        resolve_model_id(project_root, models["old"]),
        "--dataset-from-huggingface",
        _dataset_argument(config, project_root),
        "--dataset-split",
        str(dataset["evaluation"]["split"]),
        "--sample-manifest",
        str(manifest),
        "--experiment-dataset-id",
        str(dataset["id"]),
        "--model-pair-id",
        str(models["id"]),
        "--log-to-jsonl",
        str(run_dir / "results.jsonl"),
        "--log-summary-to-json",
        str(run_dir / "attack_summary.json"),
        # First-round runs persist structured artifacts locally and never
        # invoke TextAttack's legacy successful-example export/upload path.
        "--do-not-push",
        "--disable-stdout",
    ]
    _append_option(command, "--model-revision", models["new"].get("revision"))
    _append_option(
        command, "--second-model-revision", models["old"].get("revision")
    )

    search = attack.get("search")
    if search is not None:
        command.extend(["--differential-search", str(search["method"])])
        if search["method"] == "async_frontier":
            command.extend(
                [
                    "--differential-frontier-ranking",
                    str(search["ranking"]),
                    "--differential-beam-size",
                    str(search["beam_size"]),
                ]
            )
            if search["ranking"] != "dynamic":
                command.extend(
                    [
                        "--infeasible-state-policy",
                        str(search["infeasible_state_policy"]),
                    ]
                )
            if search["diagnostics"]["trace_enabled"]:
                command.extend(
                    [
                        "--search-trace-output",
                        str(run_dir / "search_trace.jsonl"),
                    ]
                )

    if attack["semantic_constraint"] == "nli":
        nli = config["semantic"]["nli"]
        command.extend(
            [
                "--nli-model-name-or-path",
                str(nli["model_name_or_path"]),
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

        threshold = config["semantic"]["threshold"]
        if threshold["source"] == "calibrated":
            command.extend(
                [
                    "--semantic-threshold-file",
                    str(_threshold_file(config, project_root)),
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
    """Store successful, failed, and skipped records beside canonical JSONL."""

    directories = {
        status: run_dir / f"{status}_examples"
        for status in ("successful", "failed", "skipped")
    }
    for directory in directories.values():
        directory.mkdir(exist_ok=True)
    records = []
    with open(run_dir / "results.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            status = row.get("result_status")
            if status not in directories:
                raise ValueError(f"Invalid result_status in JSONL: {status!r}.")
            records.append(row)
            with open(
                directories[status] / f"{row['dataset_index']}.json",
                "w",
                encoding="utf-8",
            ) as output:
                json.dump(row, output, ensure_ascii=True, indent=2, sort_keys=True)
    return records


def _augment_attack_summary(path, records):
    """Persist explicit three-way counts in the attack-stage result artifact."""

    summary = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            summary = json.load(handle)
    counts = Counter(row["result_status"] for row in records)
    summary["result_counts"] = {
        "total": len(records),
        "successful": counts["successful"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
    }
    _write_json_atomic(path, summary)


def _evaluation_command(config_path, run_dir, stage):
    """Build an independently retryable core or quality evaluator command."""

    command = [
        sys.executable,
        str(PROJECT_ROOT / "statistics" / "evaluate_improvements.py"),
        "--stage",
        stage,
        "--config",
        str(config_path),
        "--results",
        str(run_dir / "results.jsonl"),
        "--manifest",
        str(run_dir / "sample_manifest.json"),
        "--output-dir",
        str(run_dir / "metrics"),
        "--status-file",
        str(run_dir / "status.json"),
    ]
    if (run_dir / "nli_profile.json").exists():
        command.extend(["--nli-profile", str(run_dir / "nli_profile.json")])
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_experiment_config(config_path)
    validate_artifact_namespaces(config, PROJECT_ROOT)
    split = str(config["dataset"]["evaluation"]["split"])
    collection = load_dataset_collection(config["dataset"], PROJECT_ROOT)
    if split not in collection:
        raise ValueError(f"Dataset has no configured evaluation split {split!r}.")
    validate_dataset_split_schema(config["dataset"], collection[split], split)
    manifest = resolve_path(
        PROJECT_ROOT, config["dataset"]["evaluation"]["manifest"]
    )
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Test manifest does not exist: {manifest}. "
            "Run prepare_manifests.sh with this config first."
        )
    manifest_payload = validate_manifest_identity(
        manifest,
        config,
        split,
        PROJECT_ROOT,
    )

    run_dir = run_directory(config, PROJECT_ROOT)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = _prepare_run_metadata(
        run_dir, config, manifest, manifest_payload
    )

    status_path = run_dir / "status.json"
    if (run_dir / "results.jsonl").exists():
        records = _load_completed_results(run_dir, manifest_payload)
        _augment_attack_summary(run_dir / "attack_summary.json", records)
        _set_stage(status_path, "attack", "completed", resumed=True)
        print(f'results existed: {run_dir}')
    else:
        _set_stage(status_path, "attack", "running")
        try:
            # A restarted attack must not append to a trace left by a failed run.
            trace_path = run_dir / "search_trace.jsonl"
            if trace_path.exists():
                trace_path.unlink()
            subprocess.run(
                _attack_command(config, run_dir, manifest, PROJECT_ROOT),
                cwd=PROJECT_ROOT,
                check=True,
            )
            records = _split_results(run_dir)
            _augment_attack_summary(run_dir / "attack_summary.json", records)
            _set_stage(status_path, "attack", "completed", resumed=False)
        except Exception as exc:
            _set_stage(status_path, "attack", "failed", exc)
            raise

    core_config = config["evaluation"]["core"]
    query_budget = config["attack"]["query_budget"]
    if _core_evaluation_complete(run_dir, core_config, query_budget):
        _set_stage(status_path, "core_evaluation", "completed", resumed=True)
    else:
        _set_stage(status_path, "core_evaluation", "running")
        try:
            subprocess.run(
                _evaluation_command(resolved_config_path, run_dir, "core"),
                cwd=PROJECT_ROOT,
                check=True,
            )
            _set_stage(status_path, "core_evaluation", "completed", resumed=False)
        except Exception as exc:
            _set_stage(status_path, "core_evaluation", "failed", exc)
            raise

    quality_config = config["evaluation"]["quality"]
    if _quality_evaluation_complete(run_dir, quality_config):
        _set_stage(status_path, "quality_evaluation", "completed", resumed=True)
    else:
        _set_stage(status_path, "quality_evaluation", "running")
        quality_process = subprocess.run(
            _evaluation_command(resolved_config_path, run_dir, "quality"),
            cwd=PROJECT_ROOT,
            check=False,
        )
        quality_summary = run_dir / "metrics" / "quality.json"
        if quality_process.returncode != 0 or not quality_summary.exists():
            _set_stage(
                status_path,
                "quality_evaluation",
                "failed",
                f"quality evaluator exited with code {quality_process.returncode}",
            )
        else:
            quality = _read_json(quality_summary)
            _set_stage(
                status_path,
                "quality_evaluation",
                quality["status"],
                resumed=False,
            )


if __name__ == "__main__":
    main()
