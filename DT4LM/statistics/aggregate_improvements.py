#!/usr/bin/env python
"""Aggregate schema-v4 DT4LM runs into one paper-facing CSV."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("outputs/dt4lm-improvements/runs")
DEFAULT_OUTPUT = "summary.csv"
METRICS_SCHEMA_VERSION = 4
QUALITY_METRICS = ("bleu", "meteor", "rouge_l", "bertscore")

IDENTITY_COLUMNS = [
    "dataset",
    "model_pair",
    "method",
    "seed",
    "experiment_id",
    "old_model",
    "old_revision",
    "old_model_training_seed",
    "new_model",
    "new_revision",
    "new_model_training_seed",
    "recipe",
    "recipe_parameters",
    "differential_objective",
    "search_method",
    "search_algorithm",
    "frontier_ranking",
    "beam_size",
    "infeasible_state_policy",
    "query_budget",
    "manifest_split",
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
    "success_at_200",
    "success_at_300",
    "success_at_400",
    "success_at_500",
    "success_at_600",
    "success_at_700",
    "success_at_800",
    "success_at_900",
    "success_at_1000",
    "success_query_auc",
    "bpqc",
    "normalized_bpqc",
    "budget_penalized_queries_median",
    "amr",
    "model_pair_query_total",
    "model_pair_qps",
    "successful_query_count",
    "successful_queries_median",
    "successful_queries_q1",
    "successful_queries_q3",
    "recipe_diagnostic_sample_count",
    "transformation_call_total",
    "transformation_call_mean",
    "generated_candidate_total",
    "generated_candidate_mean",
    "constraint_filter_call_total",
    "constraint_filter_call_mean",
    "constraint_filter_input_total",
    "constraint_filter_input_mean",
    "constraint_passed_candidate_total",
    "constraint_passed_candidate_mean",
    "candidate_constraint_pass_rate",
    "generated_candidates_per_model_pair_query",
    "search_diagnostic_sample_count",
    "search_expansions_mean",
    "search_max_depth_mean",
    "success_path_depth_mean",
    "success_path_depth_median",
    "frontier_update_count",
    "frontier_state_slot_count",
    "frontier_size_mean",
    "frontier_size_max",
    "rank1_size_mean",
    "frontier_modified_set_diversity_mean",
    "frontier_depth_diversity_mean",
    "infeasible_fill_event_rate",
    "infeasible_retained_state_rate",
    "hard_discarded_infeasible_state_count",
    "hard_discard_rate",
    "non_top1_path_rate",
    "recover_path_count",
    "post_root_old_prediction_error_path_rate",
    "recover_first_infeasible_depth_mean",
    "recover_first_infeasible_depth_median",
    "recover_first_recovery_depth_mean",
    "recover_first_recovery_depth_median",
    "recover_depth_span_mean",
    "recover_depth_span_median",
    "duplicate_state_rate",
    "query_cache_hit_rate",
    "budget_truncation_rate",
]

QUALITY_COLUMNS = [
    "bleu",
    "meteor",
    "rouge_l",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "quality_successful_sample_count",
]

RESOURCE_COLUMNS = [
    "end_to_end_seconds",
    "end_to_end_seconds_per_success",
    "peak_vram_bytes",
    "frontier_sort_seconds",
    "frontier_sort_time_ratio",
]

PROVENANCE_COLUMNS = [
    "attack_created_at",
    "core_evaluated_at",
    "quality_evaluated_at",
    "metrics_schema_version",
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
    """Load one resolved complete experiment config."""

    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Resolved config must be a mapping: {path}")
    return payload


def _resolve_input(path):
    """Resolve an input path relative to the checked-out DT4LM root."""

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


def _model_field(config, role, field):
    """Read one field from a normalized old/new model specification."""

    return config["models"][role].get(field)


def _validate_v4(run_dir, config, core, quality, query_data, manifest):
    """Reject incomplete or internally inconsistent schema-v4 artifacts."""

    for name, payload in (
        ("core.json", core),
        ("quality.json", quality),
        ("query_data.json", query_data),
    ):
        if payload.get("schema_version") != METRICS_SCHEMA_VERSION:
            raise ValueError(
                f"{run_dir / 'metrics' / name} is not schema v4; run "
                "statistics/recompute_metrics.py first."
            )

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
        raise ValueError(f"Core attackable count is inconsistent in {run_dir}.")
    if total != manifest.get("effective_sample_size"):
        raise ValueError(f"Core total does not match the manifest in {run_dir}.")
    if core.get("query_budget") != config["attack"]["query_budget"]:
        raise ValueError(f"Core query budget does not match config in {run_dir}.")
    if query_data.get("query_budget") != core.get("query_budget"):
        raise ValueError(f"Query data budget does not match core in {run_dir}.")

    dataset = config["dataset"]
    if manifest.get("dataset_id") not in {None, dataset["id"]}:
        raise ValueError(f"Manifest dataset does not match config in {run_dir}.")
    configured_split = (dataset.get("evaluation") or {}).get("split")
    if configured_split and manifest.get("split") != configured_split:
        raise ValueError(f"Manifest split does not match config in {run_dir}.")

    data = query_data.get("data") or {}
    column_names = (
        "dataset_index",
        "result_status",
        "model_pair_queries",
        "queries_to_success",
        "budget_penalized_queries",
    )
    columns = [data.get(name) for name in column_names]
    if not all(isinstance(column, list) and len(column) == total for column in columns):
        raise ValueError(f"Query-data columns are inconsistent in {run_dir}.")
    if data["dataset_index"] != manifest.get("selected_indices"):
        raise ValueError(f"Query-data order does not match manifest in {run_dir}.")
    if query_data.get("sample_count") != total:
        raise ValueError(f"Query-data sample count is inconsistent in {run_dir}.")
    if query_data.get("successful_sample_count") != successful:
        raise ValueError(f"Query-data success count is inconsistent in {run_dir}.")

    if quality.get("successful_sample_count") != successful:
        raise ValueError(f"Quality sample count does not match core in {run_dir}.")
    if set(quality.get("metrics") or {}) != set(QUALITY_METRICS):
        raise ValueError(f"Quality artifact lacks the four metrics in {run_dir}.")


def _quality_fields(quality):
    """Flatten paper-facing quality metric values."""

    metrics = quality.get("metrics") or {}
    fields = {"quality_successful_sample_count": quality.get("successful_sample_count")}
    fields["bleu"] = ((metrics.get("bleu") or {}).get("values") or {}).get("value")
    fields["meteor"] = ((metrics.get("meteor") or {}).get("values") or {}).get(
        "value"
    )
    fields["rouge_l"] = ((metrics.get("rouge_l") or {}).get("values") or {}).get(
        "value"
    )
    bertscore = ((metrics.get("bertscore") or {}).get("values") or {})
    for name in ("precision", "recall", "f1"):
        fields[f"bertscore_{name}"] = bertscore.get(name)
    return fields


def _search_algorithm(attack):
    """Name the concrete search implementation behind a config-level method."""

    search = attack.get("search") or {"method": "legacy_greedy"}
    if search["method"] == "async_frontier":
        return "AsyncDifferentialBeamSearch"
    return {
        "kuleshov_var": "ComparatorGreedySearch",
        "leap": "LEAP",
        "faster-alzantot": "AlzantotGeneticAlgorithm",
    }.get(attack["recipe"])


def build_row(run_dir):
    """Validate and flatten one self-contained experiment run."""

    config = _read_yaml(run_dir / "config.resolved.yaml")
    core = _read_json(run_dir / "metrics" / "core.json")
    quality = _read_json(run_dir / "metrics" / "quality.json")
    query_data = _read_json(run_dir / "metrics" / "query_data.json")
    manifest = _read_json(run_dir / "sample_manifest.json")
    provenance = _read_json(run_dir / "provenance.json")
    _validate_v4(run_dir, config, core, quality, query_data, manifest)

    experiment = config["experiment"]
    attack = config["attack"]
    search = attack.get("search") or {"method": "legacy_greedy"}
    initial = core.get("initial_state_counts") or {}
    core_runtime = core.get("evaluation_runtime") or {}
    quality_runtime = quality.get("evaluation_runtime") or {}
    row = {
        "dataset": config["dataset"]["id"],
        "model_pair": config["models"]["id"],
        "method": experiment["method"],
        "seed": experiment["seed"],
        "experiment_id": experiment["id"],
        "old_model": _model_field(config, "old", "name_or_path"),
        "old_revision": _model_field(config, "old", "revision"),
        "old_model_training_seed": _model_field(config, "old", "training_seed"),
        "new_model": _model_field(config, "new", "name_or_path"),
        "new_revision": _model_field(config, "new", "revision"),
        "new_model_training_seed": _model_field(config, "new", "training_seed"),
        "recipe": attack["recipe"],
        "recipe_parameters": json.dumps(
            attack.get("recipe_parameters") or {}, sort_keys=True
        ),
        "differential_objective": attack["differential_objective"],
        "search_method": search["method"],
        "search_algorithm": _search_algorithm(attack),
        "frontier_ranking": search.get("ranking"),
        "beam_size": search.get("beam_size"),
        "infeasible_state_policy": search.get("infeasible_state_policy"),
        "query_budget": attack["query_budget"],
        "manifest_split": manifest.get("split"),
        "manifest_sample_count": manifest.get("effective_sample_size"),
        "manifest_seed": manifest.get("seed"),
        "manifest_selection_sha256": manifest.get("selection_sha256"),
        "initial_both_correct": initial.get("both_correct"),
        "initial_new_correct_old_wrong": initial.get("new_correct_old_wrong"),
        "initial_both_wrong": initial.get("both_wrong"),
        "initial_already_differential": initial.get("already_differential"),
        "attack_created_at": provenance.get("created_at"),
        "core_evaluated_at": core_runtime.get("evaluated_at"),
        "quality_evaluated_at": quality_runtime.get("evaluated_at"),
        "metrics_schema_version": core.get("schema_version"),
    }
    for name in CORE_COLUMNS:
        if not name.startswith("initial_"):
            row[name] = core.get(name)
    for name, value in core.items():
        if name.startswith("success_at_"):
            row[name] = value
    row.update(_quality_fields(quality))
    resources = core.get("resources") or {}
    row.update({name: resources.get(name) for name in RESOURCE_COLUMNS})
    return row


def _dynamic_budget_columns(rows):
    """Preserve configured Success@B values beyond the standard paper grid."""

    known = set(CORE_COLUMNS)
    dynamic = {
        key
        for row in rows
        for key in row
        if key.startswith("success_at_") and key not in known
    }
    return sorted(dynamic, key=lambda key: int(key.rsplit("_", 1)[-1]))


def write_summary(input_dir, output_path):
    """Aggregate complete runs and atomically replace one CSV."""

    run_dirs = discover_runs(input_dir)
    if not run_dirs:
        raise ValueError(f"No experiment runs found below {input_dir}.")
    rows = []
    skipped = 0
    for run_dir in run_dirs:
        try:
            rows.append(build_row(run_dir))
        except FileNotFoundError as exc:
            skipped += 1
            print(f"Skipping incomplete experiment {run_dir}: {exc}", file=sys.stderr)
    if skipped:
        print(f"Skipped {skipped} incomplete experiment(s).", file=sys.stderr)
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
