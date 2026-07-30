#!/usr/bin/env python
"""Analyze paired, stratified SemDT human judgments with bootstrap CIs."""

import argparse
from collections import defaultdict
import json
import os
import random
from statistics import mean


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _cohen_kappa(pairs):
    """Calculate binary Cohen's kappa without an external statistics package."""

    if not pairs:
        return None
    agreement = sum(left == right for left, right in pairs) / len(pairs)
    left_positive = sum(left for left, _ in pairs) / len(pairs)
    right_positive = sum(right for _, right in pairs) / len(pairs)
    chance = (
        left_positive * right_positive
        + (1 - left_positive) * (1 - right_positive)
    )
    return (agreement - chance) / (1 - chance) if chance != 1 else 1.0


def _method_estimates(observations, populations, sample_count):
    """Estimate semantic rate and ValidGSR using actual success strata."""

    by_method_stratum = defaultdict(list)
    for observation in observations:
        for method, value in observation["labels"].items():
            by_method_stratum[(method, observation["stratum"])].append(value)
    output = {}
    for method in ("Base", "SemDT"):
        numerator = 0.0
        success_total = 0
        strata = {}
        for stratum, population in populations.items():
            if method == "Base" and stratum == "semdt_only_success":
                continue
            if method == "SemDT" and stratum == "base_only_success":
                continue
            values = by_method_stratum.get((method, stratum), [])
            if population and not values:
                raise ValueError(
                    f"No human labels for non-empty {method}/{stratum}."
                )
            rate = mean(values) if values else None
            strata[stratum] = {
                "population": population,
                "sample_count": len(values),
                "semantic_preservation_rate": rate,
            }
            if values:
                numerator += population * rate
                success_total += population
        output[method] = {
            "successful_generations": success_total,
            "semantic_preservation_rate": (
                numerator / success_total if success_total else None
            ),
            "valid_gsr": numerator / sample_count,
            "strata": strata,
        }
    return output


def _percentile(values):
    values = sorted(values)
    return [
        values[int(0.025 * (len(values) - 1))],
        values[int(0.975 * (len(values) - 1))],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=765)
    parser.add_argument("--base-summary")
    parser.add_argument("--semdt-summary")
    args = parser.parse_args()

    reviews = {row["review_id"]: row for row in read_jsonl(args.reviews)}
    with open(args.key, encoding="utf-8") as handle:
        key = json.load(handle)
    observations = []
    reviewer_pairs = []
    for mapping in key["rows"]:
        row = reviews[mapping["review_id"]]
        labels = {}
        for side in ("a", "b"):
            method = mapping.get(f"candidate_{side}_method")
            if method is None:
                continue
            reviewer_left = row.get(f"reviewer_1_{side}")
            reviewer_right = row.get(f"reviewer_2_{side}")
            final = row.get(f"final_{side}")
            if type(reviewer_left) is not bool or type(reviewer_right) is not bool:
                raise ValueError(
                    f"{mapping['review_id']} is missing two boolean reviews."
                )
            reviewer_pairs.append((reviewer_left, reviewer_right))
            if final is None and reviewer_left == reviewer_right:
                final = reviewer_left
            if type(final) is not bool:
                raise ValueError(
                    f"{mapping['review_id']} needs a boolean adjudicated label."
                )
            labels[method] = final
        observations.append(
            {
                "review_id": mapping["review_id"],
                "stratum": mapping["stratum"],
                "labels": labels,
            }
        )

    populations = key["stratum_populations"]
    sample_count = int(key["manifest_sample_count"])
    estimates = _method_estimates(observations, populations, sample_count)
    by_stratum = defaultdict(list)
    for observation in observations:
        by_stratum[observation["stratum"]].append(observation)
    rng = random.Random(args.seed)
    bootstrap = {
        method: {"semantic_preservation_rate": [], "valid_gsr": []}
        for method in ("Base", "SemDT")
    }
    for _ in range(args.bootstrap_samples):
        # Resampling whole common-success rows preserves the paired judgments.
        resampled = []
        for rows in by_stratum.values():
            resampled.extend(rng.choice(rows) for _ in rows)
        iteration = _method_estimates(resampled, populations, sample_count)
        for method in bootstrap:
            for metric in bootstrap[method]:
                value = iteration[method][metric]
                if value is not None:
                    bootstrap[method][metric].append(value)
    for method in estimates:
        for metric, values in bootstrap[method].items():
            estimates[method][f"{metric}_bootstrap_95"] = _percentile(values)

    agreement = sum(left == right for left, right in reviewer_pairs) / len(
        reviewer_pairs
    )
    output = {
        "methods": estimates,
        "reviewer_agreement": agreement,
        "cohen_kappa": _cohen_kappa(reviewer_pairs),
        "bootstrap_samples": args.bootstrap_samples,
        "semdt_semantic_improvement_pp": 100
        * (
            estimates["SemDT"]["semantic_preservation_rate"]
            - estimates["Base"]["semantic_preservation_rate"]
        ),
        "semdt_valid_gsr_not_below_base": (
            estimates["SemDT"]["valid_gsr"] >= estimates["Base"]["valid_gsr"]
        ),
    }
    if args.base_summary and args.semdt_summary:
        with open(args.base_summary, encoding="utf-8") as handle:
            base_automatic = json.load(handle)
        with open(args.semdt_summary, encoding="utf-8") as handle:
            semdt_automatic = json.load(handle)
        base_qps = base_automatic["model_pair_qps"]
        semdt_qps = semdt_automatic["model_pair_qps"]
        qps_increase = (
            (semdt_qps - base_qps) / base_qps
            if base_qps not in {None, 0} and semdt_qps is not None
            else None
        )
        semantic_gain = output["semdt_semantic_improvement_pp"]
        output["semdt_expansion_decision"] = {
            "semantic_gain_at_least_5pp": semantic_gain >= 5,
            "valid_gsr_not_below_base": output[
                "semdt_valid_gsr_not_below_base"
            ],
            "qps_increase_at_most_30pct": (
                qps_increase <= 0.30 if qps_increase is not None else None
            ),
            "qps_relative_increase": qps_increase,
            "passes": (
                semantic_gain >= 5
                and output["semdt_valid_gsr_not_below_base"]
                and qps_increase is not None
                and qps_increase <= 0.30
            ),
        }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=True, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
