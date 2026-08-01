#!/usr/bin/env python
"""Create a reproducible, blinded three-stratum human-evaluation sample."""

import argparse
from collections import defaultdict
import json
import math
import os
import random


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _is_success(row):
    """Use the explicit result status while retaining readable legacy errors."""

    status = row.get("result_status")
    if status not in {"successful", "failed", "skipped"}:
        raise ValueError(f"Invalid or missing result_status: {status!r}.")
    return status == "successful"


def _allocate(groups, target):
    """Honor five-item minima, then allocate remaining slots proportionally."""

    total = sum(len(rows) for rows in groups.values())
    target = min(target, total)
    allocation = {
        name: min(5, len(rows)) for name, rows in groups.items() if rows
    }
    remaining = target - sum(allocation.values())
    if remaining < 0:
        # This is only possible for a tiny requested target; preserve strata in
        # deterministic name order instead of silently exceeding the target.
        allocation = {name: 0 for name in groups}
        for name in sorted(groups):
            for _ in groups[name]:
                if sum(allocation.values()) == target:
                    return allocation
                allocation[name] += 1
    capacities = {
        name: len(groups[name]) - allocation.get(name, 0) for name in groups
    }
    population = sum(len(rows) for rows in groups.values())
    quotas = {
        name: remaining * len(groups[name]) / population for name in groups
    }
    additions = {
        name: min(capacities[name], int(math.floor(quotas[name])))
        for name in groups
    }
    for name, count in additions.items():
        allocation[name] = allocation.get(name, 0) + count
        capacities[name] -= count
        remaining -= count
    ranking = sorted(
        groups,
        key=lambda name: (
            -(quotas[name] - math.floor(quotas[name])),
            name,
        ),
    )
    # A capped stratum can leave more than one pass of remainder slots.
    while remaining:
        progressed = False
        for name in ranking:
            if capacities[name]:
                allocation[name] = allocation.get(name, 0) + 1
                capacities[name] -= 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("Human-evaluation allocation exhausted capacity.")
    return allocation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-results", required=True)
    parser.add_argument("--semdt-results", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key-output", required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=765)
    args = parser.parse_args()

    base = {row["dataset_index"]: row for row in read_jsonl(args.base_results)}
    semdt = {row["dataset_index"]: row for row in read_jsonl(args.semdt_results)}
    if set(base) != set(semdt):
        raise ValueError("Base and SemDT must contain the same manifest indices.")
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)

    groups = defaultdict(list)
    for dataset_index in sorted(base):
        base_success = _is_success(base[dataset_index])
        semdt_success = _is_success(semdt[dataset_index])
        if base_success and semdt_success:
            groups["common_success"].append(dataset_index)
        elif base_success:
            groups["base_only_success"].append(dataset_index)
        elif semdt_success:
            groups["semdt_only_success"].append(dataset_index)
    allocation = _allocate(groups, args.sample_size)
    rng = random.Random(args.seed)
    selected = []
    for stratum in sorted(groups):
        selected.extend(
            (stratum, index)
            for index in rng.sample(groups[stratum], allocation.get(stratum, 0))
        )
    rng.shuffle(selected)

    review_rows = []
    key_rows = []
    for ordinal, (stratum, dataset_index) in enumerate(selected, start=1):
        candidates = []
        if _is_success(base[dataset_index]):
            candidates.append(("Base", base[dataset_index]["candidate_input"]))
        if _is_success(semdt[dataset_index]):
            candidates.append(("SemDT", semdt[dataset_index]["candidate_input"]))
        rng.shuffle(candidates)
        review_id = f"human-{ordinal:04d}"
        # Method names live only in the separate key artifact.
        row = {
            "review_id": review_id,
            "original_input": base[dataset_index]["original_input"],
            "ground_truth_output": base[dataset_index]["ground_truth_output"],
            "candidate_a": candidates[0][1],
            "candidate_b": candidates[1][1] if len(candidates) == 2 else None,
            "reviewer_1_a": None,
            "reviewer_2_a": None,
            "final_a": None,
            "reviewer_1_b": None,
            "reviewer_2_b": None,
            "final_b": None,
        }
        review_rows.append(row)
        key_rows.append(
            {
                "review_id": review_id,
                "dataset_index": dataset_index,
                "stratum": stratum,
                "stratum_population": len(groups[stratum]),
                "inclusion_weight": len(groups[stratum]) / allocation[stratum],
                "candidate_a_method": candidates[0][0],
                "candidate_b_method": (
                    candidates[1][0] if len(candidates) == 2 else None
                ),
            }
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in review_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    key_payload = {
        "seed": args.seed,
        "sample_count": len(review_rows),
        "manifest_sample_count": manifest["effective_sample_size"],
        "stratum_populations": {
            stratum: len(rows) for stratum, rows in groups.items()
        },
        "stratum_sample_counts": allocation,
        "rows": key_rows,
    }
    os.makedirs(os.path.dirname(args.key_output) or ".", exist_ok=True)
    with open(args.key_output, "w", encoding="utf-8") as handle:
        json.dump(key_payload, handle, ensure_ascii=True, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
