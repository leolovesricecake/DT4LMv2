"""Independent sampling and reporting for SemDT trajectory shift."""

from collections import defaultdict
import random
from typing import Dict, Iterable, List, Sequence

from .candidate_collection import _allocate_by_largest_remainder
from .threshold_search import WeightedSemanticExample, evaluate_thresholds


def trajectory_stratum(record: Dict) -> str:
    """Stratify accepted/rejected candidates across NLI score regions."""

    entailment_bin = min(4, max(0, int(float(record["entailment_score"]) * 5)))
    contradiction_bin = min(
        4, max(0, int(float(record["contradiction_score"]) * 5))
    )
    decision = "accepted" if record["accepted"] else "rejected"
    return f"{decision}-e{entailment_bin}-c{contradiction_bin}"


def sample_trajectory_audit(
    records: Sequence[Dict], *, sample_size: int = 100, seed: int = 765
) -> List[Dict]:
    """Select a fixed-size stratified audit from actual NLI checks."""

    unique = {}
    for record in records:
        unique.setdefault(record["candidate_id"], record)
    if len(unique) < sample_size:
        return sorted(
            (
                {
                    **row,
                    "trajectory_stratum": trajectory_stratum(row),
                    "inclusion_weight": 1.0,
                }
                for row in unique.values()
            ),
            key=lambda row: row["candidate_id"],
        )
    groups = defaultdict(list)
    for record in unique.values():
        groups[trajectory_stratum(record)].append(record)
    allocation = _allocate_by_largest_remainder(
        {group: len(rows) for group, rows in groups.items()}, sample_size
    )
    rng = random.Random(seed)
    selected = []
    for group in sorted(groups):
        rows = sorted(groups[group], key=lambda row: row["candidate_id"])
        count = allocation[group]
        for row in rng.sample(rows, count):
            # The inverse inclusion probability restores the actual trajectory
            # mixture when precision, recall, and acceptance are estimated.
            selected.append(
                {
                    **row,
                    "trajectory_stratum": group,
                    "inclusion_weight": len(rows) / count,
                }
            )
    return sorted(selected, key=lambda row: row["candidate_id"])


def _weighted_quantile(values, quantile):
    """Return a deterministic inverse-CDF weighted quantile."""

    ordered = sorted(values, key=lambda item: item[0])
    total = sum(weight for _, weight in ordered)
    target = quantile * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    return ordered[-1][0] if ordered else None


def score_distribution(records, entailment_threshold, contradiction_threshold):
    """Summarize one weighted NLI score population at frozen thresholds."""

    records = list(records)
    weighted_entailment = [
        (
            float(row["entailment_score"]),
            float(row.get("inclusion_weight", 1.0)),
        )
        for row in records
    ]
    weighted_contradiction = [
        (
            float(row["contradiction_score"]),
            float(row.get("inclusion_weight", 1.0)),
        )
        for row in records
    ]
    total_weight = sum(weight for _, weight in weighted_entailment)
    accepted_weight = sum(
        float(row.get("inclusion_weight", 1.0))
        for row in records
        if float(row["entailment_score"]) >= entailment_threshold
        and float(row["contradiction_score"]) <= contradiction_threshold
    )

    def summarize(values):
        if not values:
            return {"mean": None, "q10": None, "median": None, "q90": None}
        total = sum(weight for _, weight in values)
        return {
            "mean": sum(value * weight for value, weight in values) / total,
            "q10": _weighted_quantile(values, 0.10),
            "median": _weighted_quantile(values, 0.50),
            "q90": _weighted_quantile(values, 0.90),
        }

    return {
        "sample_count": len(records),
        "population_weight": total_weight,
        "entailment": summarize(weighted_entailment),
        "contradiction": summarize(weighted_contradiction),
        "frozen_threshold_acceptance_rate": (
            accepted_weight / total_weight if total_weight else None
        ),
    }


def weighted_ks_distance(left_records, right_records, score_key):
    """Compute a weighted empirical two-sample Kolmogorov-Smirnov distance."""

    left = [
        (float(row[score_key]), float(row.get("inclusion_weight", 1.0)))
        for row in left_records
    ]
    right = [
        (float(row[score_key]), float(row.get("inclusion_weight", 1.0)))
        for row in right_records
    ]
    if not left or not right:
        return None
    left_total = sum(weight for _, weight in left)
    right_total = sum(weight for _, weight in right)
    boundaries = sorted({value for value, _ in left + right})
    distance = 0.0
    for boundary in boundaries:
        left_cdf = (
            sum(weight for value, weight in left if value <= boundary)
            / left_total
        )
        right_cdf = (
            sum(weight for value, weight in right if value <= boundary)
            / right_total
        )
        distance = max(distance, abs(left_cdf - right_cdf))
    return distance


def distribution_shift_report(
    base_records,
    trajectory_records,
    *,
    entailment_threshold,
    contradiction_threshold,
):
    """Compare score populations without deriving or modifying thresholds."""

    base_records = list(base_records)
    trajectory_records = list(trajectory_records)
    return {
        "base": score_distribution(
            base_records, entailment_threshold, contradiction_threshold
        ),
        "trajectory": score_distribution(
            trajectory_records, entailment_threshold, contradiction_threshold
        ),
        "weighted_ks_entailment": weighted_ks_distance(
            base_records, trajectory_records, "entailment_score"
        ),
        "weighted_ks_contradiction": weighted_ks_distance(
            base_records, trajectory_records, "contradiction_score"
        ),
        "threshold_changed": False,
    }


def audit_fixed_threshold(
    trajectory_records: Iterable[Dict],
    labels_by_candidate_id: Dict[str, bool],
    *,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> Dict:
    """Evaluate the frozen threshold; this function cannot return a new one."""

    examples = []
    accepted_weight = total_weight = 0.0
    records = list(trajectory_records)
    for record in records:
        candidate_id = record["candidate_id"]
        if candidate_id not in labels_by_candidate_id:
            continue
        weight = float(record.get("inclusion_weight", 1.0))
        total_weight += weight
        accepted_weight += weight * int(record["accepted"])
        examples.append(
            WeightedSemanticExample(
                candidate_id=candidate_id,
                entailment_score=float(record["entailment_score"]),
                contradiction_score=float(record["contradiction_score"]),
                semantic_preserved=bool(labels_by_candidate_id[candidate_id]),
                weight=weight,
            )
        )
    metrics = evaluate_thresholds(
        examples, entailment_threshold, contradiction_threshold
    )
    return {
        "sample_count": len(examples),
        "observed_acceptance_rate": (
            accepted_weight / total_weight if total_weight else None
        ),
        "precision": metrics.precision,
        "recall": metrics.recall,
        "entailment_threshold": entailment_threshold,
        "contradiction_threshold": contradiction_threshold,
        "threshold_changed": False,
    }


def judge_agreement(left: Dict[str, bool], right: Dict[str, bool]) -> Dict:
    """Report agreement only on candidate IDs shared by two judge runs."""

    shared = sorted(set(left) & set(right))
    agreements = sum(left[candidate_id] == right[candidate_id] for candidate_id in shared)
    return {
        "shared_candidates": len(shared),
        "agreements": agreements,
        "agreement_rate": agreements / len(shared) if shared else None,
    }
