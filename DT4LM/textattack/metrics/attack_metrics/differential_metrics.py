"""Metrics with manifest-defined denominators for DT4LM experiments."""

from typing import Any, Dict, Iterable, Optional

from textattack.metrics.metric import Metric


def _is_success(result) -> bool:
    return result.__class__.__name__ == "SuccessfulAttackResult"


def _modification_rate(result) -> float:
    original = result.original_result.attacked_text
    perturbed = result.perturbed_result.attacked_text
    if original.num_words == 0:
        return 0.0
    return len(perturbed.attack_attrs.get("modified_indices", set())) / original.num_words


def calculate_differential_metrics(
    records: Iterable[Dict[str, Any]],
    *,
    sample_count: int,
    eligible_count: Optional[int] = None,
    test_split_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Calculate paper-compatible query metrics from structured records."""

    records = list(records)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if len(records) != sample_count:
        raise ValueError(
            f"Expected {sample_count} result records, received {len(records)}."
        )

    successes = [record for record in records if record["success"]]
    success_count = len(successes)
    total_queries = sum(int(record["model_pair_queries"]) for record in records)
    metrics = {
        "sample_count": sample_count,
        "successful_generations": success_count,
        "perturbation_induced_gsr": success_count / sample_count,
        "model_pair_query_total": total_queries,
        # JSON null represents the plan's N/A value when no generation succeeds.
        "model_pair_qps": (
            total_queries / success_count if success_count else None
        ),
        "success_at_100": sum(
            record["success"] and int(record["model_pair_queries"]) <= 100
            for record in records
        )
        / sample_count,
        "success_at_500": sum(
            record["success"] and int(record["model_pair_queries"]) <= 500
            for record in records
        )
        / sample_count,
        "success_at_1000": sum(
            record["success"] and int(record["model_pair_queries"]) <= 1000
            for record in records
        )
        / sample_count,
        "amr": (
            sum(float(record["modification_rate"]) for record in successes)
            / success_count
            if success_count
            else None
        ),
    }
    if eligible_count is not None and test_split_size is not None:
        if test_split_size <= 0 or not 0 <= eligible_count <= test_split_size:
            raise ValueError("Eligibility counts are inconsistent.")
        metrics["eligibility_rate"] = eligible_count / test_split_size
    return metrics


class DifferentialMetrics(Metric):
    """Calculate differential metrics from TextAttack result objects."""

    def __init__(self, manifest=None):
        self.manifest = manifest

    def calculate(self, results):
        records = [
            {
                "success": _is_success(result),
                "model_pair_queries": result.num_queries,
                "modification_rate": (
                    _modification_rate(result) if _is_success(result) else 0.0
                ),
            }
            for result in results
        ]
        sample_count = (
            self.manifest.sample_count if self.manifest is not None else len(records)
        )
        return calculate_differential_metrics(
            records,
            sample_count=sample_count,
            eligible_count=(
                self.manifest.eligible_count
                if self.manifest is not None
                else None
            ),
            test_split_size=(
                self.manifest.test_split_size
                if self.manifest is not None
                else None
            ),
        )
