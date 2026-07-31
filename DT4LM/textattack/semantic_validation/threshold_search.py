"""Weighted threshold search and frozen-set evaluation for SemDT."""

from dataclasses import asdict, dataclass
import json
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class WeightedSemanticExample:
    """One successfully judged candidate used by threshold evaluation."""

    candidate_id: str
    entailment_score: float
    contradiction_score: float
    semantic_preserved: bool
    weight: float = 1.0


@dataclass(frozen=True)
class ThresholdMetrics:
    precision: Optional[float]
    recall: Optional[float]
    accepted_weight: float
    true_positive_weight: float
    false_positive_weight: float
    false_negative_weight: float
    true_negative_weight: float


@dataclass(frozen=True)
class ThresholdArtifact:
    """Frozen threshold selection plus its search-set evidence."""

    entailment_threshold: float
    contradiction_threshold: float
    min_precision: float
    search_metrics: ThresholdMetrics
    judge_backend: str
    judge_model: str
    dataset: str
    split_manifest_hash: str
    threshold_search_method: str
    threshold_step: float
    nli_config: Optional[Dict[str, Any]] = None
    model_pair_id: Optional[str] = None

    def save(self, path: str) -> None:
        payload = asdict(self)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        # Historical artifacts predate explicit search provenance.
        data.setdefault("threshold_search_method", "grid")
        data.setdefault("threshold_step", 0.01)
        data.setdefault("model_pair_id", None)
        data["search_metrics"] = ThresholdMetrics(**data["search_metrics"])
        return cls(**data)


def evaluate_thresholds(
    examples: Iterable[WeightedSemanticExample],
    entailment_threshold: float,
    contradiction_threshold: float,
) -> ThresholdMetrics:
    """Evaluate one closed-boundary NLI acceptance rule."""

    tp = fp = fn = tn = 0.0
    for example in examples:
        accepted = (
            example.entailment_score >= entailment_threshold
            and example.contradiction_score <= contradiction_threshold
        )
        if accepted and example.semantic_preserved:
            tp += example.weight
        elif accepted:
            fp += example.weight
        elif example.semantic_preserved:
            fn += example.weight
        else:
            tn += example.weight
    accepted_weight = tp + fp
    positive_weight = tp + fn
    return ThresholdMetrics(
        precision=tp / accepted_weight if accepted_weight else None,
        recall=tp / positive_weight if positive_weight else None,
        accepted_weight=accepted_weight,
        true_positive_weight=tp,
        false_positive_weight=fp,
        false_negative_weight=fn,
        true_negative_weight=tn,
    )


def search_thresholds(
    examples: Sequence[WeightedSemanticExample],
    *,
    min_precision: float = 0.95,
    step: float = 0.01,
) -> Tuple[float, float, ThresholdMetrics]:
    """Maximize recall subject to weighted precision on the search set."""

    if not examples:
        raise ValueError("Threshold search requires judged examples.")
    if not 0 <= min_precision <= 1:
        raise ValueError("min_precision must lie in [0, 1].")
    steps = round(1.0 / step)
    if step <= 0 or abs(steps * step - 1.0) > 1e-9:
        raise ValueError("step must divide the interval [0, 1] exactly.")

    best = None
    for entailment_index in range(steps + 1):
        entailment_threshold = round(entailment_index * step, 10)
        for contradiction_index in range(steps + 1):
            contradiction_threshold = round(contradiction_index * step, 10)
            metrics = evaluate_thresholds(
                examples, entailment_threshold, contradiction_threshold
            )
            if (
                metrics.precision is None
                or metrics.precision < min_precision
            ):
                continue
            # The first three fields implement the preregistered tie-break.
            key = (
                metrics.recall if metrics.recall is not None else -1.0,
                metrics.precision,
                metrics.accepted_weight,
                entailment_threshold,
                -contradiction_threshold,
            )
            if best is None or key > best[0]:
                best = (
                    key,
                    entailment_threshold,
                    contradiction_threshold,
                    metrics,
                )
    if best is None:
        raise ValueError(
            "No threshold pair satisfies the configured precision floor."
        )
    return best[1], best[2], best[3]


def validation_report(
    examples: Sequence[WeightedSemanticExample],
    entailment_threshold: float,
    contradiction_threshold: float,
    *,
    bootstrap_samples: int = 10000,
    seed: int = 765,
) -> Dict:
    """Report frozen-threshold metrics and percentile bootstrap intervals."""

    point = evaluate_thresholds(
        examples, entailment_threshold, contradiction_threshold
    )
    rng = random.Random(seed)
    precision_samples: List[float] = []
    recall_samples: List[float] = []
    for _ in range(bootstrap_samples):
        resampled = [rng.choice(examples) for _ in examples]
        metric = evaluate_thresholds(
            resampled, entailment_threshold, contradiction_threshold
        )
        if metric.precision is not None:
            precision_samples.append(metric.precision)
        if metric.recall is not None:
            recall_samples.append(metric.recall)

    def interval(values):
        if not values:
            return [None, None]
        values = sorted(values)
        return [
            values[int(0.025 * (len(values) - 1))],
            values[int(0.975 * (len(values) - 1))],
        ]

    return {
        "metrics": asdict(point),
        "precision_bootstrap_95": interval(precision_samples),
        "recall_bootstrap_95": interval(recall_samples),
        "bootstrap_samples": bootstrap_samples,
    }


def needs_supplemental_audit(
    validation_examples: Sequence[WeightedSemanticExample],
    *,
    minimum_positive: int = 100,
) -> bool:
    """Trigger a separate audit without modifying the frozen 800/200 split."""

    return (
        sum(example.semantic_preserved for example in validation_examples)
        < minimum_positive
    )
