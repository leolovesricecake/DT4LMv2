#!/usr/bin/env python
"""Recompute DT4LM improvement metrics from immutable JSON artifacts."""

import argparse
from collections import Counter
import json
import math
import os
from statistics import mean


def read_jsonl(path):
    """Read a JSONL artifact without importing the full TextAttack package."""

    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten_fields(fields):
    """Preserve structured field order while producing quality-metric text."""

    return " ".join(str(value) for value in fields.values())


def core_metrics(records, manifest):
    """Apply the preregistered manifest and paper-compatible query formulas."""

    sample_count = int(manifest["sample_count"])
    if len(records) != sample_count:
        raise ValueError(
            f"Manifest expects {sample_count} rows, received {len(records)}."
        )
    successful = [row for row in records if row["success"]]
    success_count = len(successful)
    query_total = sum(int(row["model_pair_queries"]) for row in records)
    result = {
        "sample_count": sample_count,
        "successful_generations": success_count,
        "perturbation_induced_gsr": success_count / sample_count,
        "model_pair_query_total": query_total,
        # JSON null is the machine-readable N/A value for a zero denominator.
        "model_pair_qps": query_total / success_count if success_count else None,
        "amr": (
            mean(float(row["modification_rate"]) for row in successful)
            if successful
            else None
        ),
    }
    for budget in (100, 500, 1000):
        result[f"success_at_{budget}"] = (
            sum(
                row["success"] and int(row["model_pair_queries"]) <= budget
                for row in records
            )
            / sample_count
        )
    eligible_count = int(manifest["eligible_count"])
    test_split_size = int(manifest["test_split_size"])
    result["eligibility_rate"] = eligible_count / test_split_size
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


def quality_metrics(successful, *, include_bertscore=True):
    """Calculate quality only on generated differential successes."""

    if not successful:
        return {
            "bleu": None,
            "meteor": None,
            "rouge_l": None,
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
        }
    references = [
        flatten_fields(row["original_input"]) for row in successful
    ]
    candidates = [
        flatten_fields(row["candidate_input"]) for row in successful
    ]
    reference_tokens = [text.split() for text in references]
    candidate_tokens = [text.split() for text in candidates]
    quality = {
        "rouge_l": mean(
            _rouge_l(reference, candidate)
            for reference, candidate in zip(reference_tokens, candidate_tokens)
        )
    }
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        smoothing = SmoothingFunction().method1
        quality["bleu"] = mean(
            sentence_bleu([reference], candidate, smoothing_function=smoothing)
            for reference, candidate in zip(reference_tokens, candidate_tokens)
        )
    except Exception as exc:
        quality["bleu"] = None
        quality["bleu_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from nltk.translate.meteor_score import meteor_score

        quality["meteor"] = mean(
            meteor_score([reference], candidate)
            for reference, candidate in zip(reference_tokens, candidate_tokens)
        )
    except Exception as exc:
        quality["meteor"] = None
        quality["meteor_error"] = f"{type(exc).__name__}: {exc}"
    if include_bertscore:
        try:
            from bert_score import score

            precision, recall, f1 = score(
                candidates, references, lang="en", verbose=False
            )
            quality.update(
                {
                    "bertscore_precision": float(precision.mean()),
                    "bertscore_recall": float(recall.mean()),
                    "bertscore_f1": float(f1.mean()),
                }
            )
        except Exception as exc:
            quality.update(
                {
                    "bertscore_precision": None,
                    "bertscore_recall": None,
                    "bertscore_f1": None,
                    "bertscore_error": f"{type(exc).__name__}: {exc}",
                }
            )
    return quality


def resource_metrics(records, profile=None):
    """Keep victim-query and NLI/endpoint resource costs visibly separate."""

    success_count = sum(row["success"] for row in records)
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


def compare_to_base(summary, base):
    """Report the predeclared absolute and relative comparisons."""

    comparisons = {}
    for key in ("perturbation_induced_gsr", "amr", "model_pair_qps"):
        current = summary.get(key)
        reference = base.get(key)
        comparisons[f"{key}_absolute_difference"] = (
            current - reference
            if current is not None and reference is not None
            else None
        )
        comparisons[f"{key}_relative_difference"] = (
            (current - reference) / reference
            if current is not None and reference not in {None, 0}
            else None
        )
    comparisons["gsr_equivalent_within_1pp"] = (
        abs(comparisons["perturbation_induced_gsr_absolute_difference"]) <= 0.01
        if comparisons["perturbation_induced_gsr_absolute_difference"] is not None
        else None
    )
    for key in ("amr", "model_pair_qps"):
        value = comparisons[f"{key}_relative_difference"]
        comparisons[f"{key}_not_worse_than_5pct"] = (
            value <= 0.05 if value is not None else None
        )
    current_time = summary.get("resources", {}).get(
        "end_to_end_seconds_per_success"
    )
    base_time = base.get("resources", {}).get("end_to_end_seconds_per_success")
    comparisons["end_to_end_time_ratio"] = (
        current_time / base_time
        if current_time is not None and base_time not in {None, 0}
        else None
    )
    return comparisons


def evaluate(args):
    records = read_jsonl(args.results)
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    summary = core_metrics(records, manifest)
    summary["method"] = args.method
    successful = [row for row in records if row["success"]]
    summary["quality"] = quality_metrics(
        successful, include_bertscore=not args.skip_bertscore
    )
    profile = None
    if args.nli_profile and os.path.exists(args.nli_profile):
        with open(args.nli_profile, encoding="utf-8") as handle:
            profile = json.load(handle)
    summary["resources"] = resource_metrics(records, profile)
    if args.base_summary:
        with open(args.base_summary, encoding="utf-8") as handle:
            summary["comparison_to_base"] = compare_to_base(
                summary, json.load(handle)
            )
        if args.method == "lexidt":
            comparison = summary["comparison_to_base"]
            gsr_difference = comparison[
                "perturbation_induced_gsr_absolute_difference"
            ]
            qps_difference = comparison["model_pair_qps_relative_difference"]
            amr_difference = comparison["amr_relative_difference"]
            summary["lexidt_expansion_decision"] = {
                "criterion_1_gsr_gain_with_cost_equivalence": (
                    gsr_difference is not None
                    and gsr_difference >= 0.02
                    and qps_difference is not None
                    and qps_difference <= 0.05
                    and amr_difference is not None
                    and amr_difference <= 0.05
                ),
                "criterion_2_equivalent_gsr_lower_qps": (
                    gsr_difference is not None
                    and abs(gsr_difference) <= 0.01
                    and qps_difference is not None
                    and qps_difference <= -0.10
                ),
                "criterion_3_equivalent_gsr_lower_amr": (
                    gsr_difference is not None
                    and abs(gsr_difference) <= 0.01
                    and amr_difference is not None
                    and amr_difference <= -0.10
                ),
            }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nli-profile")
    parser.add_argument("--base-summary")
    parser.add_argument("--method")
    parser.add_argument("--skip-bertscore", action="store_true")
    args = parser.parse_args()
    summary = evaluate(args)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
