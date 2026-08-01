#!/usr/bin/env python
"""Evaluate one DT4LM run without comparing it to another experiment."""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean, median, quantiles
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dt4lm_artifacts import resolve_model_id  # noqa: E402
from improvement_config import load_experiment_config  # noqa: E402


RESULT_STATUSES = frozenset(("successful", "failed", "skipped"))
INITIAL_STATES = frozenset(
    ("both_correct", "new_correct_old_wrong", "both_wrong", "already_differential")
)


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


def core_metrics(records, manifest, *, success_budgets, query_budget):
    """Compute count, denominator, and query metrics from immutable records."""

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
        raise ValueError("Result counts do not satisfy total=successful+failed+skipped.")

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
    success_query_data = []
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
                raise ValueError("queries_to_success exceeds recorded queries or budget.")
            success_queries.append(query_to_success)
            success_query_data.append(
                {
                    "dataset_index": int(row["dataset_index"]),
                    "queries_to_success": query_to_success,
                }
            )
        elif query_to_success is not None:
            raise ValueError("Failed and skipped rows require null queries_to_success.")

    q1, q3 = _quartiles(success_queries)
    result = {
        "schema_version": 2,
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
        "query_budget": int(query_budget),
        "success_query_data": success_query_data,
        "successful_query_counts": success_queries,
        "successful_queries_median": median(success_queries) if success_queries else None,
        "successful_queries_q1": q1,
        "successful_queries_q3": q3,
    }
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
    return result


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


def run_quality_metrics(records, quality_config, output_dir, project_root):
    """Run and persist quality metrics independently on successful generations."""

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
        }

    calculators = {
        "bleu": lambda: empty_or(bleu),
        "meteor": lambda: empty_or(meteor),
        "rouge_l": lambda: empty_or(rouge_l),
        "bertscore": lambda: empty_or(bertscore),
    }
    output_dir = Path(output_dir)
    results = {}
    for name, calculate in calculators.items():
        payload = _metric_payload(quality_config[name], calculate)
        _write_json_atomic(output_dir / f"{name}.json", payload)
        results[name] = payload["status"]
    summary = {
        "status": "failed" if "failed" in results.values() else "completed",
        "metrics": results,
        "successful_sample_count": len(successful),
    }
    _write_json_atomic(output_dir / "quality.json", summary)
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
    core = core_metrics(
        records,
        manifest,
        success_budgets=core_config["success_budgets"],
        query_budget=config["attack"]["query_budget"],
    )
    profile = None
    if args.nli_profile and Path(args.nli_profile).exists():
        with open(args.nli_profile, encoding="utf-8") as handle:
            profile = json.load(handle)
    output_dir = Path(args.output_dir)
    _write_json_atomic(output_dir / "core.json", core)
    _write_json_atomic(output_dir / "resources.json", resource_metrics(records, profile))


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
