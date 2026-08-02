#!/usr/bin/env python
"""Evaluate one DT4LM run without comparing it to another experiment."""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
from statistics import mean, median, quantiles
import sys

from packaging.version import Version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dt4lm_artifacts import resolve_model_id  # noqa: E402
from improvement_config import load_experiment_config  # noqa: E402


RESULT_STATUSES = frozenset(("successful", "failed", "skipped"))
METRICS_SCHEMA_VERSION = 3
INITIAL_STATES = frozenset(
    ("both_correct", "new_correct_old_wrong", "both_wrong", "already_differential")
)


def evaluation_runtime():
    """Capture the environment that produced a metric artifact."""

    packages = {}
    for package in ("torch", "transformers", "datasets", "bert-score"):
        try:
            packages[package] = package_version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "packages": packages,
    }


def read_jsonl(path):
    """Read a JSONL artifact without importing the full TextAttack package."""

    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json_atomic(path, payload):
    """Replace one metric artifact only after its complete JSON is serialized."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _update_stage_status(path, stage, status, error=None):
    """Optionally keep a run's stage state in sync during manual retries."""

    if not path:
        return
    path = Path(path)
    payload = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    entry = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        entry["error"] = str(error)
    payload[stage] = entry
    _write_json_atomic(path, payload)


def flatten_fields(fields):
    """Preserve structured field order while producing quality-metric text."""

    return " ".join(str(value) for value in fields.values())


def _record_status(row):
    """Require the new explicit status instead of inferring skipped as failure."""

    status = row.get("result_status")
    if status not in RESULT_STATUSES:
        raise ValueError(f"Invalid or missing result_status: {status!r}.")
    return status


def _quartiles(values):
    """Return inclusive quartiles while handling a single successful sample."""

    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    first, _, third = quantiles(values, n=4, method="inclusive")
    return first, third


def _search_diagnostic_metrics(records, statuses):
    """Aggregate optional AE-PBS diagnostics without rejecting legacy rows."""

    diagnostics = [
        row.get("search_diagnostics")
        for row, status in zip(records, statuses)
        if status != "skipped" and isinstance(row.get("search_diagnostics"), dict)
    ]
    successful_diagnostics = [
        row.get("search_diagnostics")
        for row, status in zip(records, statuses)
        if status == "successful"
        and isinstance(row.get("search_diagnostics"), dict)
    ]
    adaptive = [
        item for item in diagnostics if item.get("epsilon_mode") == "adaptive"
    ]
    adaptive_initialized = [
        item
        for item in adaptive
        if item.get("epsilon_zero_initialization") is not None
    ]

    def values(name, source=diagnostics):
        return [
            float(item[name])
            for item in source
            if item.get(name) is not None
        ]

    def average(name, source=diagnostics):
        observed = values(name, source)
        return mean(observed) if observed else None

    duplicate_states = sum(
        int(item.get("duplicate_state_count") or 0) for item in diagnostics
    )
    constraint_passed = sum(
        int(item.get("constraint_passed_candidate_count") or 0)
        for item in diagnostics
    )
    cache_hits = sum(
        int(item.get("query_cache_hit_count") or 0) for item in diagnostics
    )
    cache_misses = sum(
        int(item.get("query_cache_miss_count") or 0) for item in diagnostics
    )
    truncated = sum(
        int(item.get("budget_truncated_candidate_count") or 0)
        for item in diagnostics
    )
    non_top1 = [
        item.get("root_dynamic_rank") > 1
        for item in successful_diagnostics
        if item.get("root_dynamic_rank") is not None
    ]
    escaped = [
        bool(item.get("path_has_negative_old_margin"))
        for item in successful_diagnostics
        if item.get("path_has_negative_old_margin") is not None
    ]
    old_prediction_errors = [
        bool(item.get("path_has_old_prediction_error"))
        for item in successful_diagnostics
        if item.get("path_has_old_prediction_error") is not None
    ]
    post_root_escaped = [
        bool(item.get("path_has_post_root_negative_old_margin"))
        for item in successful_diagnostics
        if item.get("path_has_post_root_negative_old_margin") is not None
    ]
    post_root_old_prediction_errors = [
        bool(item.get("path_has_post_root_old_prediction_error"))
        for item in successful_diagnostics
        if item.get("path_has_post_root_old_prediction_error") is not None
    ]
    discarded_infeasible_states = sum(
        int(item.get("discarded_infeasible_state_count") or 0)
        for item in diagnostics
    )
    candidate_states = sum(
        int(item.get("candidate_state_count") or 0) for item in diagnostics
    )
    epsilon_ratios = values("epsilon_to_root_margin_ratio", adaptive)
    initialization_expansions = values(
        "epsilon_initialization_expansion", adaptive
    )
    return {
        "search_diagnostic_sample_count": len(diagnostics),
        "search_expansions_mean": average("expansion_count"),
        "search_max_depth_mean": average("max_depth"),
        "frontier_size_mean": average("frontier_size_mean"),
        "frontier_size_max": (
            max(values("frontier_size_max"))
            if values("frontier_size_max")
            else None
        ),
        "rank1_size_mean": average("rank1_size_mean"),
        "frontier_modified_set_diversity_mean": average(
            "frontier_modified_set_diversity_mean"
        ),
        "frontier_depth_diversity_mean": average(
            "frontier_depth_diversity_mean"
        ),
        "duplicate_state_rate": (
            duplicate_states / constraint_passed if constraint_passed else None
        ),
        "query_cache_hit_rate": (
            cache_hits / (cache_hits + cache_misses)
            if cache_hits + cache_misses
            else None
        ),
        "budget_truncation_rate": (
            truncated / cache_misses if cache_misses else None
        ),
        "non_top1_path_rate": mean(non_top1) if non_top1 else None,
        "escape_path_rate": mean(escaped) if escaped else None,
        "root_inclusive_escape_path_rate": (
            mean(escaped) if escaped else None
        ),
        "old_prediction_error_path_rate": (
            mean(old_prediction_errors) if old_prediction_errors else None
        ),
        "root_inclusive_old_prediction_error_path_rate": (
            mean(old_prediction_errors) if old_prediction_errors else None
        ),
        "post_root_escape_path_rate": (
            mean(post_root_escaped) if post_root_escaped else None
        ),
        "post_root_old_prediction_error_path_rate": (
            mean(post_root_old_prediction_errors)
            if post_root_old_prediction_errors
            else None
        ),
        "discarded_infeasible_state_count": discarded_infeasible_states,
        "discarded_infeasible_state_rate": (
            discarded_infeasible_states / candidate_states
            if candidate_states
            else None
        ),
        "epsilon_zero_initialization_rate": (
            mean(
                bool(item.get("epsilon_zero_initialization"))
                for item in adaptive_initialized
            )
            if adaptive_initialized
            else None
        ),
        "epsilon_to_root_margin_ratio_median": (
            median(epsilon_ratios) if epsilon_ratios else None
        ),
        "epsilon_initialization_expansion_mean": (
            mean(initialization_expansions) if initialization_expansions else None
        ),
    }


def core_metrics(records, manifest, *, success_budgets, query_budget):
    """Compute core metrics and compact per-success query columns."""

    expected_indices = list(manifest["selected_indices"])
    observed_indices = [int(row["dataset_index"]) for row in records]
    if observed_indices != expected_indices:
        raise ValueError(
            "Result dataset indices must exactly match manifest order; "
            f"expected {expected_indices!r}, got {observed_indices!r}."
        )

    statuses = [_record_status(row) for row in records]
    counts = Counter(statuses)
    total = int(manifest["effective_sample_size"])
    if len(records) != total:
        raise ValueError(f"Manifest expects {total} rows, received {len(records)}.")
    successful = [
        row for row, status in zip(records, statuses) if status == "successful"
    ]
    success_count = counts["successful"]
    failed_count = counts["failed"]
    skipped_count = counts["skipped"]
    attackable_count = success_count + failed_count
    if total != success_count + failed_count + skipped_count:
        raise ValueError(
            "Result counts do not satisfy total=successful+failed+skipped."
        )

    initial_states = Counter()
    for row, status in zip(records, statuses):
        state = row.get("initial_state")
        if state not in INITIAL_STATES:
            raise ValueError(f"Invalid or missing initial_state: {state!r}.")
        if status == "skipped" and state != "already_differential":
            raise ValueError("Only already_differential samples may be skipped.")
        initial_states[state] += 1

    total_queries = 0
    success_queries = []
    budget_penalized_queries = []
    success_query_columns = {"dataset_index": [], "queries_to_success": []}
    for row, status in zip(records, statuses):
        queries = int(row["model_pair_queries"])
        if queries <= 0:
            raise ValueError("model_pair_queries must be positive for every sample.")
        total_queries += queries
        query_to_success = row.get("queries_to_success")
        if status == "successful":
            if not isinstance(query_to_success, int) or query_to_success <= 0:
                raise ValueError("Successful rows require positive queries_to_success.")
            if query_to_success > queries or query_to_success > query_budget:
                raise ValueError(
                    "queries_to_success exceeds recorded queries or budget."
                )
            success_queries.append(query_to_success)
            budget_penalized_queries.append(query_to_success)
            # Columnar storage avoids repeating field names for every success.
            success_query_columns["dataset_index"].append(
                int(row["dataset_index"])
            )
            success_query_columns["queries_to_success"].append(query_to_success)
        elif query_to_success is not None:
            raise ValueError("Failed and skipped rows require null queries_to_success.")
        elif status == "failed":
            # Penalize exhausted and early-frontier failures equally at the
            # configured budget when comparing success and query efficiency.
            budget_penalized_queries.append(int(query_budget))

    q1, q3 = _quartiles(success_queries)
    successful_nli = [
        row["nli"]
        for row in successful
        if isinstance(row.get("nli"), dict)
        and row["nli"].get("entailment_score") is not None
        and row["nli"].get("contradiction_score") is not None
    ]
    result = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "total": total,
        "successful": success_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "attackable": attackable_count,
        "paper_gsr": (
            success_count / attackable_count if attackable_count else None
        ),
        "sample_generation_rate": success_count / total,
        "preexisting_differential_rate": skipped_count / total,
        "initial_state_counts": {
            state: initial_states.get(state, 0) for state in sorted(INITIAL_STATES)
        },
        "model_pair_query_total": total_queries,
        # QPS deliberately includes queries from successful, failed, and skipped rows.
        "model_pair_qps": total_queries / success_count if success_count else None,
        "amr": (
            mean(float(row["modification_rate"]) for row in successful)
            if successful
            else None
        ),
        "successful_nli_sample_count": len(successful_nli),
        "successful_nli_entailment_mean": (
            mean(float(item["entailment_score"]) for item in successful_nli)
            if successful_nli
            else None
        ),
        "successful_nli_contradiction_mean": (
            mean(float(item["contradiction_score"]) for item in successful_nli)
            if successful_nli
            else None
        ),
        "successful_nli_acceptance_rate": (
            mean(bool(item.get("accepted")) for item in successful_nli)
            if successful_nli
            else None
        ),
        "query_budget": int(query_budget),
        "successful_query_count": len(success_queries),
        "successful_queries_median": (
            median(success_queries) if success_queries else None
        ),
        "successful_queries_q1": q1,
        "successful_queries_q3": q3,
        "budget_penalized_query_count": len(budget_penalized_queries),
        "budget_penalized_queries_mean": (
            mean(budget_penalized_queries) if budget_penalized_queries else None
        ),
        "budget_penalized_queries_median": (
            median(budget_penalized_queries) if budget_penalized_queries else None
        ),
    }
    result.update(_search_diagnostic_metrics(records, statuses))
    for budget in success_budgets:
        numerator = sum(value <= budget for value in success_queries)
        result[f"success_at_{budget}"] = (
            numerator / attackable_count if attackable_count else None
        )
        result[f"sample_success_at_{budget}"] = numerator / total

    # Each success at query q contributes to C(b) for q <= b <= query_budget.
    result["sq_auc"] = (
        sum(query_budget - value + 1 for value in success_queries)
        / (attackable_count * query_budget)
        if attackable_count
        else None
    )
    success_query_artifact = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "successful_sample_count": len(success_queries),
        "data": success_query_columns,
    }
    return result, success_query_artifact


def _lcs_length(left, right):
    """Compute token LCS with one rolling row for ROUGE-L."""

    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _rouge_l(reference, candidate):
    """Calculate token-level ROUGE-L F1 for one reference/candidate pair."""

    if not reference or not candidate:
        return 0.0
    overlap = _lcs_length(reference, candidate)
    precision = overlap / len(candidate)
    recall = overlap / len(reference)
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


@contextmanager
def _offline_huggingface(enabled):
    """Temporarily force local-only Hugging Face resolution for BERTScore."""

    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    if enabled:
        for name in names:
            os.environ[name] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _metric_payload(config, function):
    """Capture one optional quality metric's value or isolated failure."""

    if not config["enabled"]:
        return {"status": "disabled", "config": config, "values": None}
    try:
        values = function()
        return {"status": "completed", "config": config, "values": values}
    except Exception as exc:
        return {
            "status": "failed",
            "config": config,
            "values": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _local_model_weight_format(model_path, torch_version=None):
    """Validate local model weights against Transformers' secure-load policy."""

    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"BERTScore local model directory does not exist: {model_path}."
        )

    # Transformers prefers safetensors when either a monolithic or sharded
    # checkpoint is present, so no pickle security gate is involved.
    safe_weights = list(model_path.glob("*.safetensors"))
    safe_index = model_path / "model.safetensors.index.json"
    if safe_weights or safe_index.is_file():
        return "safetensors"

    pickle_weights = list(model_path.glob("pytorch_model*.bin"))
    pickle_index = model_path / "pytorch_model.bin.index.json"
    if pickle_weights or pickle_index.is_file():
        installed = torch_version
        if installed is None:
            try:
                installed = package_version("torch")
            except PackageNotFoundError as exc:
                raise RuntimeError(
                    "BERTScore found pickle .bin weights, but torch is not installed."
                ) from exc
        if Version(installed) < Version("2.6.0"):
            raise RuntimeError(
                "BERTScore model uses pickle .bin weights, but Transformers blocks "
                f"torch.load with torch {installed} because of CVE-2025-32434. "
                "Upgrade to torch>=2.6 (CUDA 11.8: pip install torch==2.6.0 "
                "torchvision==0.21.0 torchaudio==2.6.0 --index-url "
                "https://download.pytorch.org/whl/cu118), or configure an "
                "equivalent checkpoint containing model.safetensors. Do not "
                "disable or patch around the Transformers security check."
            )
        return "pytorch_bin"

    raise FileNotFoundError(
        "BERTScore local model has no *.safetensors or pytorch_model*.bin weights: "
        f"{model_path}."
    )


def run_quality_metrics(records, quality_config, output_dir, project_root):
    """Persist all independently evaluated quality metrics in one artifact."""

    successful = [row for row in records if _record_status(row) == "successful"]
    references = [flatten_fields(row["original_input"]) for row in successful]
    candidates = [flatten_fields(row["candidate_input"]) for row in successful]
    reference_tokens = [text.split() for text in references]
    candidate_tokens = [text.split() for text in candidates]

    def empty_or(calculate):
        return calculate() if successful else {"value": None, "sample_count": 0}

    def bleu():
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        smoothing = SmoothingFunction().method1
        value = mean(
            sentence_bleu([reference], candidate, smoothing_function=smoothing)
            for reference, candidate in zip(reference_tokens, candidate_tokens)
        )
        return {"value": value, "sample_count": len(successful)}

    def meteor():
        from nltk.translate.meteor_score import meteor_score

        value = mean(
            meteor_score([reference], candidate)
            for reference, candidate in zip(reference_tokens, candidate_tokens)
        )
        return {"value": value, "sample_count": len(successful)}

    def rouge_l():
        value = mean(
            _rouge_l(reference, candidate)
            for reference, candidate in zip(reference_tokens, candidate_tokens)
        )
        return {"value": value, "sample_count": len(successful)}

    def bertscore():
        metric_config = quality_config["bertscore"]
        model = resolve_model_id(project_root, metric_config["model_name_or_path"])
        offline = not metric_config["allow_remote_download"]
        if offline and not Path(model).exists():
            raise FileNotFoundError(
                "BERTScore local model does not exist while remote downloads are "
                f"disabled: {model}."
            )
        weight_format = None
        if Path(model).exists():
            weight_format = _local_model_weight_format(model)
        from bert_score import score

        kwargs = {
            "model_type": model,
            "num_layers": int(metric_config["num_layers"]),
            "device": metric_config.get("device"),
            "batch_size": int(metric_config["batch_size"]),
            "idf": bool(metric_config["idf"]),
            "rescale_with_baseline": bool(
                metric_config["rescale_with_baseline"]
            ),
            "verbose": False,
        }
        baseline = metric_config.get("baseline_path")
        if baseline:
            baseline_path = Path(baseline).expanduser()
            if not baseline_path.is_absolute():
                baseline_path = project_root / baseline_path
            kwargs["baseline_path"] = str(baseline_path.resolve())
        with _offline_huggingface(offline):
            precision, recall, f1 = score(candidates, references, **kwargs)
        return {
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "f1": float(f1.mean()),
            "sample_count": len(successful),
            "model_weight_format": weight_format,
        }

    calculators = {
        "bleu": lambda: empty_or(bleu),
        "meteor": lambda: empty_or(meteor),
        "rouge_l": lambda: empty_or(rouge_l),
        "bertscore": lambda: empty_or(bertscore),
    }
    output_dir = Path(output_dir)
    quality_path = output_dir / "quality.json"
    previous_metrics = {}
    if quality_path.exists():
        try:
            with open(quality_path, encoding="utf-8") as handle:
                previous = json.load(handle)
            if (
                previous.get("schema_version") == METRICS_SCHEMA_VERSION
                and previous.get("successful_sample_count") == len(successful)
            ):
                previous_metrics = previous.get("metrics") or {}
        except (json.JSONDecodeError, OSError):
            previous_metrics = {}

    results = {}
    for name, calculate in calculators.items():
        prior = previous_metrics.get(name) or {}
        expected_status = (
            "completed" if quality_config[name]["enabled"] else "disabled"
        )
        # A completed metric with identical config is a valid quality checkpoint.
        if (
            prior.get("status") == expected_status
            and prior.get("config") == quality_config[name]
        ):
            payload = prior
        else:
            payload = _metric_payload(quality_config[name], calculate)
        results[name] = payload
    summary = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "status": (
            "failed"
            if any(item["status"] == "failed" for item in results.values())
            else "completed"
        ),
        "metrics": results,
        "successful_sample_count": len(successful),
        "evaluation_runtime": evaluation_runtime(),
    }
    _write_json_atomic(quality_path, summary)
    # Remove schema-v2 files only after the consolidated artifact is durable.
    for name in calculators:
        legacy_path = output_dir / f"{name}.json"
        if legacy_path.exists():
            legacy_path.unlink()
    return summary


def resource_metrics(records, profile=None):
    """Keep target-model and NLI resource costs visibly separate."""

    success_count = sum(_record_status(row) == "successful" for row in records)
    wall_seconds = sum(
        float(row.get("wall_clock_seconds") or 0.0) for row in records
    )
    resources = {
        "end_to_end_seconds": wall_seconds,
        "end_to_end_seconds_per_success": (
            wall_seconds / success_count if success_count else None
        ),
        "peak_vram_bytes": max(
            (int(row.get("peak_vram_bytes") or 0) for row in records),
            default=0,
        ),
    }
    if profile:
        resources["nli"] = profile
    return resources


def run_core(args, config, records):
    """Persist network-independent metrics before optional quality evaluation."""

    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    core_config = config["evaluation"]["core"]
    core, success_queries = core_metrics(
        records,
        manifest,
        success_budgets=core_config["success_budgets"],
        query_budget=config["attack"]["query_budget"],
    )
    profile = None
    if args.nli_profile and Path(args.nli_profile).exists():
        with open(args.nli_profile, encoding="utf-8") as handle:
            profile = json.load(handle)
    core["resources"] = resource_metrics(records, profile)
    core["evaluation_runtime"] = evaluation_runtime()
    output_dir = Path(args.output_dir)
    _write_json_atomic(output_dir / "core.json", core)
    _write_json_atomic(output_dir / "success_queries.json", success_queries)
    legacy_resources = output_dir / "resources.json"
    if legacy_resources.exists():
        legacy_resources.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("core", "quality", "all"), default="all")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nli-profile")
    parser.add_argument("--status-file")
    args = parser.parse_args()

    config = load_experiment_config(Path(args.config).resolve())
    records = read_jsonl(args.results)
    if args.stage in {"core", "all"}:
        _update_stage_status(
            args.status_file, "core_evaluation", "running"
        )
        try:
            run_core(args, config, records)
            _update_stage_status(
                args.status_file, "core_evaluation", "completed"
            )
        except Exception as exc:
            _update_stage_status(
                args.status_file, "core_evaluation", "failed", exc
            )
            raise
    if args.stage in {"quality", "all"}:
        _update_stage_status(
            args.status_file, "quality_evaluation", "running"
        )
        try:
            quality = run_quality_metrics(
                records,
                config["evaluation"]["quality"],
                args.output_dir,
                PROJECT_ROOT,
            )
            _update_stage_status(
                args.status_file, "quality_evaluation", quality["status"]
            )
        except Exception as exc:
            _update_stage_status(
                args.status_file, "quality_evaluation", "failed", exc
            )
            raise


if __name__ == "__main__":
    main()
