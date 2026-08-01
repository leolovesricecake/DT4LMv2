"""Core differential metrics with explicit success/failure/skip counts."""

from collections import Counter
from typing import Any, Dict, Iterable

from textattack.metrics.metric import Metric


def _result_status(result) -> str:
    """Map a TextAttack result object onto the three-state experiment schema."""

    class_name = result.__class__.__name__
    if class_name == "SuccessfulAttackResult":
        return "successful"
    if class_name == "SkippedAttackResult":
        return "skipped"
    return "failed"


def _modification_rate(result) -> float:
    """Calculate word modification rate only for successful generations."""

    original = result.original_result.attacked_text
    perturbed = result.perturbed_result.attacked_text
    if original.num_words == 0:
        return 0.0
    return len(perturbed.attack_attrs.get("modified_indices", set())) / original.num_words


def calculate_differential_metrics(
    records: Iterable[Dict[str, Any]],
    *,
    sample_count: int,
) -> Dict[str, Any]:
    """Calculate count and QPS metrics from structured three-state records."""

    records = list(records)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if len(records) != sample_count:
        raise ValueError(
            f"Expected {sample_count} result records, received {len(records)}."
        )
    statuses = [record["result_status"] for record in records]
    if any(status not in {"successful", "failed", "skipped"} for status in statuses):
        raise ValueError("Every record requires a valid result_status.")
    counts = Counter(statuses)
    successes = [
        record
        for record, status in zip(records, statuses)
        if status == "successful"
    ]
    success_count = counts["successful"]
    attackable = success_count + counts["failed"]
    total_queries = sum(int(record["model_pair_queries"]) for record in records)
    return {
        "total": sample_count,
        "successful": success_count,
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "attackable": attackable,
        "paper_gsr": success_count / attackable if attackable else None,
        "sample_generation_rate": success_count / sample_count,
        "model_pair_query_total": total_queries,
        # This confirmed QPS denominator remains the number of successes.
        "model_pair_qps": total_queries / success_count if success_count else None,
        "amr": (
            sum(float(record["modification_rate"]) for record in successes)
            / success_count
            if success_count
            else None
        ),
    }


class DifferentialMetrics(Metric):
    """Calculate three-state differential metrics from TextAttack results."""

    def __init__(self, manifest=None):
        self.manifest = manifest

    def calculate(self, results):
        records = []
        for result in results:
            status = _result_status(result)
            records.append(
                {
                    "result_status": status,
                    "model_pair_queries": result.num_queries,
                    "modification_rate": (
                        _modification_rate(result) if status == "successful" else 0.0
                    ),
                }
            )
        sample_count = (
            self.manifest.sample_count if self.manifest is not None else len(records)
        )
        return calculate_differential_metrics(records, sample_count=sample_count)
