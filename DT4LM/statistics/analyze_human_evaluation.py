#!/usr/bin/env python
"""Analyze blinded DT4LM-Kuleshov/FF-PBS judgments and intervals."""

import argparse
from collections import defaultdict
import json
import os
import random
from statistics import mean


def read_jsonl(path):
    """Read one completed human-review JSONL file."""

    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _cohen_kappa(pairs):
    """Calculate binary Cohen's kappa without an external dependency."""

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


def _percentile(values):
    """Return a deterministic empirical 95% interval."""

    values = sorted(values)
    if not values:
        return None
    return [
        values[int(0.025 * (len(values) - 1))],
        values[int(0.975 * (len(values) - 1))],
    ]


def _rates(observations):
    """Calculate LPR, SPR, and their joint HVR for one uniform sample."""

    if not observations:
        return {"lpr": None, "spr": None, "hvr": None, "sample_count": 0}
    return {
        "lpr": mean(item["label_preserved"] for item in observations),
        "spr": mean(item["semantic_preserved"] for item in observations),
        "hvr": mean(
            item["label_preserved"] and item["semantic_preserved"]
            for item in observations
        ),
        "sample_count": len(observations),
    }


def _bootstrap_rates(observations, *, samples, rng):
    """Bootstrap all three human-validity rates by review unit."""

    if not observations:
        return {"lpr": None, "spr": None, "hvr": None}
    replicates = {"lpr": [], "spr": [], "hvr": []}
    for _ in range(samples):
        draw = [rng.choice(observations) for _ in observations]
        rates = _rates(draw)
        for name in replicates:
            replicates[name].append(rates[name])
    return {name: _percentile(values) for name, values in replicates.items()}


def _load_core(path, expected_total):
    """Load schema-v4 GSR needed to estimate ValidGSR."""

    with open(path, encoding="utf-8") as handle:
        core = json.load(handle)
    if core.get("schema_version") != 4:
        raise ValueError(f"Human analysis requires schema-v4 core metrics: {path}.")
    if core.get("total") != expected_total:
        raise ValueError("Core metrics and human-evaluation manifest sizes differ.")
    return core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--kuleshov-core", required=True)
    parser.add_argument("--ffpbs-core", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=765)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")

    reviews = {row["review_id"]: row for row in read_jsonl(args.reviews)}
    with open(args.key, encoding="utf-8") as handle:
        key = json.load(handle)
    if key.get("schema_version") != 2:
        raise ValueError("Human-evaluation key must use schema version 2.")

    observations_by_cohort = defaultdict(list)
    label_pairs = []
    semantic_pairs = []
    for mapping in key["rows"]:
        row = reviews[mapping["review_id"]]
        values = {}
        for dimension in ("label_preserved", "semantic_preserved"):
            left = row.get(f"reviewer_1_{dimension}")
            right = row.get(f"reviewer_2_{dimension}")
            final = row.get(f"final_{dimension}")
            if type(left) is not bool or type(right) is not bool:
                raise ValueError(
                    f"{mapping['review_id']} lacks two boolean {dimension} reviews."
                )
            if final is None and left == right:
                final = left
            if type(final) is not bool:
                raise ValueError(
                    f"{mapping['review_id']} needs adjudicated {dimension}."
                )
            values[dimension] = final
            (label_pairs if dimension == "label_preserved" else semantic_pairs).append(
                (left, right)
            )
        observation = {
            "review_id": mapping["review_id"],
            "label_preserved": values["label_preserved"],
            "semantic_preserved": values["semantic_preserved"],
        }
        for cohort in mapping["cohorts"]:
            observations_by_cohort[cohort].append(observation)

    total = int(key["manifest_sample_count"])
    cores = {
        "DT4LM-Kuleshov": _load_core(args.kuleshov_core, total),
        "FF-PBS": _load_core(args.ffpbs_core, total),
    }
    method_cohorts = {
        "DT4LM-Kuleshov": "kuleshov_overall",
        "FF-PBS": "ffpbs_overall",
    }
    rng = random.Random(args.seed)
    methods = {}
    for method, cohort in method_cohorts.items():
        observations = observations_by_cohort[cohort]
        rates = _rates(observations)
        intervals = _bootstrap_rates(
            observations, samples=args.bootstrap_samples, rng=rng
        )
        gsr = cores[method]["paper_gsr"]
        valid_gsr_replicates = []
        if observations and gsr is not None:
            for _ in range(args.bootstrap_samples):
                draw = [rng.choice(observations) for _ in observations]
                valid_gsr_replicates.append(gsr * _rates(draw)["hvr"])
        methods[method] = {
            **rates,
            "lpr_bootstrap_95": intervals["lpr"],
            "spr_bootstrap_95": intervals["spr"],
            "hvr_bootstrap_95": intervals["hvr"],
            "paper_gsr": gsr,
            "valid_gsr": gsr * rates["hvr"] if rates["hvr"] is not None else None,
            "valid_gsr_bootstrap_95": _percentile(valid_gsr_replicates),
        }

    unique_observations = observations_by_cohort["ffpbs_unique"]
    unique_rates = _rates(unique_observations)
    unique_intervals = _bootstrap_rates(
        unique_observations, samples=args.bootstrap_samples, rng=rng
    )
    output = {
        "schema_version": 2,
        "methods": methods,
        "ffpbs_unique": {
            "sample_count": unique_rates["sample_count"],
            "ivr": unique_rates["hvr"],
            "ivr_bootstrap_95": unique_intervals["hvr"],
        },
        "agreement": {
            "label_preserved_cohen_kappa": _cohen_kappa(label_pairs),
            "semantic_preserved_cohen_kappa": _cohen_kappa(semantic_pairs),
        },
        "bootstrap_samples": args.bootstrap_samples,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=True, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
