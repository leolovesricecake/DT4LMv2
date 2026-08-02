#!/usr/bin/env python
"""Aggregate self-contained DT4LM run artifacts into one paper-ready CSV."""

import argparse
import csv
import json
import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("output/dt4lm-improvements/run")
DEFAULT_OUTPUT = "summary.csv"
QUALITY_METRICS = ("bleu", "meteor", "rouge_l", "bertscore")

IDENTITY_COLUMNS = [
    "dataset",
    "model_pair",
    "method",
    "seed",
    "attack_seed",
    "experiment_id",
    "old_model",
    "old_revision",
    "old_model_training_seed",
    "new_model",
    "new_revision",
    "new_model_training_seed",
    "recipe",
    "differential_objective",
    "search_method",
    "frontier_ranking",
    "beam_size",
    "epsilon_mode",
    "epsilon_initial_quantile",
    "epsilon_initialization_max_expansions",
    "epsilon_decay",
    "semantic_constraint",
    "threshold_source",
    "threshold_backend",
    "threshold_entailment",
    "threshold_contradiction",
    "query_budget",
    "manifest_split",
    "manifest_dataset_fingerprint",
    "manifest_dataset_revision",
    "manifest_sampling_algorithm",
    "manifest_population_size",
    "manifest_requested_sample_size",
    "manifest_sample_count",
    "manifest_seed",
    "manifest_selection_sha256",
]

CORE_COLUMNS = [
    "total",
    "attackable",
    "successful",
    "failed",
    "skipped",
    "initial_both_correct",
    "initial_new_correct_old_wrong",
    "initial_both_wrong",
    "initial_already_differential",
    "paper_gsr",
    "sample_generation_rate",
    "preexisting_differential_rate",
    "success_at_100",
    "success_at_500",
    "success_at_1000",
    "sample_success_at_100",
    "sample_success_at_500",
    "sample_success_at_1000",
    "sq_auc",
    "amr",
    "model_pair_query_total",
    "model_pair_qps",
    "successful_query_count",
    "successful_queries_median",
    "successful_queries_q1",
    "successful_queries_q3",
    "budget_penalized_query_count",
    "budget_penalized_queries_mean",
    "budget_penalized_queries_median",
    "search_diagnostic_sample_count",
    "search_expansions_mean",
    "search_max_depth_mean",
    "frontier_size_mean",
    "frontier_size_max",
    "rank1_size_mean",
    "frontier_modified_set_diversity_mean",
    "frontier_depth_diversity_mean",
    "duplicate_state_rate",
    "query_cache_hit_rate",
    "budget_truncation_rate",
    "non_top1_path_rate",
    "escape_path_rate",
    "old_prediction_error_path_rate",
    "epsilon_zero_initialization_rate",
    "epsilon_to_root_margin_ratio_median",
    "epsilon_initialization_expansion_mean",
    "successful_nli_sample_count",
    "successful_nli_entailment_mean",
    "successful_nli_contradiction_mean",
    "successful_nli_acceptance_rate",
]

QUALITY_COLUMNS = [
    "bleu",
    "meteor",
    "rouge_l",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "quality_successful_sample_count",
    "quality_bleu_status",
    "quality_meteor_status",
    "quality_rouge_l_status",
    "quality_bertscore_status",
    "quality_bertscore_model",
    "quality_bertscore_num_layers",
    "quality_bertscore_idf",
    "quality_bertscore_rescale_with_baseline",
    "quality_bertscore_weight_format",
    "quality_bleu_error",
    "quality_meteor_error",
    "quality_rouge_l_error",
    "quality_bertscore_error",
]

RESOURCE_COLUMNS = [
    "end_to_end_seconds",
    "end_to_end_seconds_per_success",
    "peak_vram_bytes",
    "nli_candidates",
    "nli_directional_pairs",
    "nli_logical_directional_pairs",
    "nli_batches",
    "nli_cache_hits",
    "nli_cache_misses",
    "nli_cache_hit_rate",
    "nli_truncated_candidates",
    "nli_truncated_candidate_rate",
    "nli_truncated_directional_pairs",
    "nli_truncated_directional_pair_rate",
    "nli_inference_seconds",
    "nli_seconds_per_candidate",
    "nli_peak_vram_bytes",
]

PROVENANCE_COLUMNS = [
    "attack_status",
    "core_status",
    "quality_status",
    "git_commit",
    "git_dirty",
    "attack_created_at",
    "attack_python_version",
    "attack_torch_version",
    "attack_transformers_version",
    "core_evaluated_at",
    "quality_evaluated_at",
    "python_version",
    "torch_version",
    "transformers_version",
    "datasets_version",
    "bert_score_version",
    "cuda_version",
    "gpus",
    "metrics_schema_version",
    "run_dir",
    "success_queries_file",
]


def _read_json(path):
    """Load required JSON with a path-specific parse error."""

    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        hint = (
            " Run statistics/recompute_metrics.py first."
            if path.parent.name == "metrics"
            else ""
        )
        raise FileNotFoundError(
            f"Required run artifact is missing: {path}.{hint}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact {path}: {exc}") from exc


def _read_yaml(path):
    """Load one resolved experiment config."""

    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Resolved config must be a mapping: {path}")
    return payload


def _resolve_input(path):
    """Resolve input relative to the checked-out DT4LM root."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _output_path(input_dir, filename):
    """Keep the requested output filename directly under the input directory."""

    output = Path(filename)
    if (
        output.is_absolute()
        or output.parent != Path(".")
        or output.name in {"", ".", ".."}
    ):
        raise ValueError("--o must be a filename, not an absolute or nested path.")
    return input_dir / output.name


def discover_runs(input_dir):
    """Discover run directories in deterministic experiment order."""

    return sorted(path.parent for path in Path(input_dir).rglob("config.resolved.yaml"))


def _model_value(config, key):
    """Read a normalized or legacy model config value."""

    value = config["models"][key]
    return value.get("name_or_path") if isinstance(value, dict) else value


def _model_revision(config, key):
    """Read an optional model revision from a normalized config."""

    value = config["models"][key]
    return value.get("revision") if isinstance(value, dict) else None


def _model_training_seed(config, key):
    """Read optional training-seed provenance from a normalized model spec."""

    value = config["models"][key]
    return value.get("training_seed") if isinstance(value, dict) else None


def _stage_status(status, stage):
    """Return one pipeline-stage state without assuming status.json is complete."""

    return (status.get(stage) or {}).get("status")


def _validate_v3(run_dir, config, core, quality, queries, manifest):
    """Reject stale or internally inconsistent metrics before CSV generation."""

    for name, payload in (
        ("core.json", core),
        ("quality.json", quality),
        ("success_queries.json", queries),
    ):
        if payload.get("schema_version") != 3:
            raise ValueError(
                f"{run_dir / 'metrics' / name} is not schema v3; run "
                "statistics/recompute_metrics.py first."
            )
    if core.get("total") != manifest.get("effective_sample_size"):
        raise ValueError(f"Core total does not match sample manifest in {run_dir}.")
    total = core.get("total")
    successful = core.get("successful")
    failed = core.get("failed")
    skipped = core.get("skipped")
    if any(
        not isinstance(value, int)
        for value in (total, successful, failed, skipped)
    ):
        raise ValueError(f"Core counts must be integers in {run_dir}.")
    if total != successful + failed + skipped:
        raise ValueError(f"Core counts do not satisfy N=S+F+K in {run_dir}.")
    if core.get("attackable") != successful + failed:
        raise ValueError(f"Core attackable count does not satisfy A=S+F in {run_dir}.")
    if core.get("query_budget") != config["attack"]["query_budget"]:
        raise ValueError(f"Core query budget does not match config in {run_dir}.")
    dataset = config["dataset"]
    if manifest.get("dataset_id") not in {None, dataset["id"]}:
        raise ValueError(f"Manifest dataset does not match config in {run_dir}.")
    configured_split = (dataset.get("evaluation") or {}).get("split")
    if configured_split and manifest.get("split") != configured_split:
        raise ValueError(f"Manifest split does not match config in {run_dir}.")
    data = queries.get("data") or {}
    indices = data.get("dataset_index")
    query_counts = data.get("queries_to_success")
    expected = core.get("successful")
    if not (
        isinstance(indices, list)
        and isinstance(query_counts, list)
        and len(indices) == len(query_counts) == expected
        and queries.get("successful_sample_count") == expected
        and core.get("successful_query_count") == expected
    ):
        raise ValueError(f"Success-query columns are inconsistent in {run_dir}.")
    selected = set(manifest.get("selected_indices") or [])
    if selected and (len(set(indices)) != len(indices) or not set(indices) <= selected):
        raise ValueError(f"Success-query indices do not belong to manifest in {run_dir}.")
    query_budget = core["query_budget"]
    if any(
        not isinstance(value, int) or value <= 0 or value > query_budget
        for value in query_counts
    ):
        raise ValueError(f"Success-query values exceed the budget in {run_dir}.")
    if quality.get("successful_sample_count") != expected:
        raise ValueError(f"Quality sample count does not match successes in {run_dir}.")
    if set(quality.get("metrics") or {}) != set(QUALITY_METRICS):
        raise ValueError(f"Quality artifact does not contain four metrics in {run_dir}.")


def _quality_fields(quality):
    """Flatten quality values, statuses, and actionable failures."""

    metrics = quality.get("metrics") or {}
    fields = {"quality_successful_sample_count": quality.get("successful_sample_count")}
    for name in QUALITY_METRICS:
        metric = metrics.get(name) or {}
        fields[f"quality_{name}_status"] = metric.get("status")
        fields[f"quality_{name}_error"] = metric.get("error")
    fields["bleu"] = ((metrics.get("bleu") or {}).get("values") or {}).get("value")
    fields["meteor"] = ((metrics.get("meteor") or {}).get("values") or {}).get("value")
    fields["rouge_l"] = (
        (metrics.get("rouge_l") or {}).get("values") or {}
    ).get("value")
    bertscore = ((metrics.get("bertscore") or {}).get("values") or {})
    for name in ("precision", "recall", "f1"):
        fields[f"bertscore_{name}"] = bertscore.get(name)
    fields["quality_bertscore_weight_format"] = bertscore.get("model_weight_format")
    bertscore_config = (metrics.get("bertscore") or {}).get("config") or {}
    fields["quality_bertscore_model"] = bertscore_config.get("model_name_or_path")
    fields["quality_bertscore_num_layers"] = bertscore_config.get("num_layers")
    fields["quality_bertscore_idf"] = bertscore_config.get("idf")
    fields["quality_bertscore_rescale_with_baseline"] = bertscore_config.get(
        "rescale_with_baseline"
    )
    return fields


def _resource_fields(resources):
    """Flatten target-model and optional NLI resource profiles separately."""

    fields = {
        name: resources.get(name)
        for name in (
            "end_to_end_seconds",
            "end_to_end_seconds_per_success",
            "peak_vram_bytes",
        )
    }
    nli = resources.get("nli") or {}
    for name in RESOURCE_COLUMNS[3:]:
        fields[name] = nli.get(name.removeprefix("nli_"))
    return fields


def build_row(run_dir, input_dir):
    """Validate and flatten one self-contained experiment run."""

    config = _read_yaml(run_dir / "config.resolved.yaml")
    core = _read_json(run_dir / "metrics" / "core.json")
    quality = _read_json(run_dir / "metrics" / "quality.json")
    queries = _read_json(run_dir / "metrics" / "success_queries.json")
    manifest = _read_json(run_dir / "sample_manifest.json")
    status = _read_json(run_dir / "status.json")
    provenance = _read_json(run_dir / "provenance.json")
    _validate_v3(run_dir, config, core, quality, queries, manifest)

    experiment = config["experiment"]
    attack = config["attack"]
    search = attack.get("search") or {"method": "legacy_greedy"}
    epsilon = search.get("epsilon") or {}
    threshold = (config.get("semantic") or {}).get("threshold") or {}
    calibration = config.get("calibration") or {}
    judge = calibration.get("judge") or {}
    attack_packages = provenance.get("packages") or {}
    core_runtime = core.get("evaluation_runtime") or {}
    quality_runtime = quality.get("evaluation_runtime") or {}
    metric_runtime = quality_runtime or core_runtime
    packages = metric_runtime.get("packages") or attack_packages
    initial = core.get("initial_state_counts") or {}
    row = {
        "dataset": config["dataset"]["id"],
        "model_pair": config["models"]["id"],
        "method": experiment["method"],
        "seed": experiment["seed"],
        "attack_seed": experiment["seed"],
        "experiment_id": experiment["id"],
        "old_model": _model_value(config, "old"),
        "old_revision": _model_revision(config, "old"),
        "old_model_training_seed": _model_training_seed(config, "old"),
        "new_model": _model_value(config, "new"),
        "new_revision": _model_revision(config, "new"),
        "new_model_training_seed": _model_training_seed(config, "new"),
        "recipe": attack["recipe"],
        "differential_objective": attack["differential_objective"],
        "search_method": search["method"],
        "frontier_ranking": search.get("ranking"),
        "beam_size": search.get("beam_size"),
        "epsilon_mode": epsilon.get("mode"),
        "epsilon_initial_quantile": epsilon.get("initial_quantile"),
        "epsilon_initialization_max_expansions": epsilon.get(
            "initialization_max_expansions"
        ),
        "epsilon_decay": epsilon.get("decay"),
        "semantic_constraint": attack["semantic_constraint"],
        "threshold_source": threshold.get("source"),
        "threshold_backend": threshold.get("backend") or judge.get("backend"),
        "threshold_entailment": threshold.get("entailment"),
        "threshold_contradiction": threshold.get("contradiction"),
        "query_budget": attack["query_budget"],
        "manifest_split": manifest.get("split"),
        "manifest_dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "manifest_dataset_revision": manifest.get("dataset_revision"),
        "manifest_sampling_algorithm": manifest.get("sampling_algorithm"),
        "manifest_population_size": manifest.get("population_size"),
        "manifest_requested_sample_size": manifest.get("requested_sample_size"),
        "manifest_sample_count": manifest.get("effective_sample_size"),
        "manifest_seed": manifest.get("seed"),
        "manifest_selection_sha256": manifest.get("selection_sha256"),
        "initial_both_correct": initial.get("both_correct"),
        "initial_new_correct_old_wrong": initial.get("new_correct_old_wrong"),
        "initial_both_wrong": initial.get("both_wrong"),
        "initial_already_differential": initial.get("already_differential"),
        "attack_status": _stage_status(status, "attack"),
        "core_status": _stage_status(status, "core_evaluation"),
        "quality_status": quality.get("status"),
        "git_commit": provenance.get("git_commit"),
        "git_dirty": provenance.get("git_dirty"),
        "attack_created_at": provenance.get("created_at"),
        "attack_python_version": provenance.get("python"),
        "attack_torch_version": attack_packages.get("torch"),
        "attack_transformers_version": attack_packages.get("transformers"),
        "core_evaluated_at": core_runtime.get("evaluated_at"),
        "quality_evaluated_at": quality_runtime.get("evaluated_at"),
        "python_version": metric_runtime.get("python") or provenance.get("python"),
        "torch_version": packages.get("torch"),
        "transformers_version": packages.get("transformers"),
        "datasets_version": packages.get("datasets"),
        "bert_score_version": packages.get("bert-score"),
        "cuda_version": provenance.get("cuda_version"),
        "gpus": " | ".join(provenance.get("gpus") or []),
        "metrics_schema_version": core.get("schema_version"),
        "run_dir": str(run_dir.relative_to(input_dir)),
        "success_queries_file": str(
            (run_dir / "metrics" / "success_queries.json").relative_to(input_dir)
        ),
    }
    for name in CORE_COLUMNS:
        if not name.startswith("initial_"):
            row[name] = core.get(name)
    for name, value in core.items():
        if name.startswith("success_at_") or name.startswith("sample_success_at_"):
            row[name] = value
    row.update(_quality_fields(quality))
    row.update(_resource_fields(core.get("resources") or {}))
    nli_profile = (core.get("resources") or {}).get("nli") or {}
    row["threshold_entailment"] = (
        row["threshold_entailment"]
        if row["threshold_entailment"] is not None
        else nli_profile.get("entailment_threshold")
    )
    row["threshold_contradiction"] = (
        row["threshold_contradiction"]
        if row["threshold_contradiction"] is not None
        else nli_profile.get("contradiction_threshold")
    )
    return row


def _dynamic_budget_columns(rows):
    """Preserve nonstandard configured Success@B metrics in the CSV."""

    known = set(CORE_COLUMNS)
    dynamic = {
        key
        for row in rows
        for key in row
        if (key.startswith("success_at_") or key.startswith("sample_success_at_"))
        and key not in known
    }
    return sorted(dynamic, key=lambda key: (int(key.rsplit("_", 1)[-1]), key))


def write_summary(input_dir, output_path):
    """Aggregate all discovered runs and atomically replace one CSV."""

    run_dirs = discover_runs(input_dir)
    if not run_dirs:
        raise ValueError(f"No experiment runs found below {input_dir}.")
    rows = [build_row(run_dir, input_dir) for run_dir in run_dirs]
    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["model_pair"],
            row["method"],
            row["seed"],
        )
    )
    columns = (
        IDENTITY_COLUMNS
        + CORE_COLUMNS
        + _dynamic_budget_columns(rows)
        + QUALITY_COLUMNS
        + RESOURCE_COLUMNS
        + PROVENANCE_COLUMNS
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_path)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate DT4LM experiment metrics.")
    parser.add_argument("--i", default=str(DEFAULT_INPUT), help="Run tree root.")
    parser.add_argument("--o", default=DEFAULT_OUTPUT, help="CSV filename under --i.")
    args = parser.parse_args()

    input_dir = _resolve_input(args.i)
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Experiment input directory does not exist: {input_dir}"
        )
    output_path = _output_path(input_dir, args.o)
    count = write_summary(input_dir, output_path)
    print(f"Wrote {count} experiment row(s) to {output_path}.")


if __name__ == "__main__":
    main()
