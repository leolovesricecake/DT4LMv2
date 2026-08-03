#!/usr/bin/env python
"""Create blinded DT4LM-Kuleshov/FF-PBS human-evaluation samples."""

import argparse
import json
import os
import random


VALID_STATUSES = frozenset(("successful", "failed", "skipped"))


def read_jsonl(path):
    """Read one result JSONL artifact."""

    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _index_results(path):
    """Validate statuses and index one run by manifest dataset index."""

    indexed = {}
    for row in read_jsonl(path):
        if row.get("schema_version") != 4:
            raise ValueError(f"Human sampling requires schema-v4 rows in {path}.")
        status = row.get("result_status")
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid or missing result_status: {status!r}.")
        dataset_index = int(row["dataset_index"])
        if dataset_index in indexed:
            raise ValueError(f"Duplicate dataset index {dataset_index} in {path}.")
        indexed[dataset_index] = row
    return indexed


def _success_indices(indexed):
    """Return sorted successful dataset indices for uniform sampling."""

    return sorted(
        index
        for index, row in indexed.items()
        if row["result_status"] == "successful"
    )


def build_sample(
    kuleshov,
    ffpbs,
    *,
    method_sample_size=100,
    unique_sample_size=50,
    seed=765,
):
    """Build blinded review rows and a separate method/cohort key."""

    if set(kuleshov) != set(ffpbs):
        raise ValueError(
            "DT4LM-Kuleshov and FF-PBS must contain identical manifest indices."
        )
    if method_sample_size <= 0 or unique_sample_size < 0:
        raise ValueError("Sample sizes must be positive, with unique size non-negative.")

    rng = random.Random(seed)
    kuleshov_success = _success_indices(kuleshov)
    ff_success = _success_indices(ffpbs)
    ff_unique = sorted(set(ff_success) - set(kuleshov_success))
    selected = {
        "kuleshov_overall": set(
            rng.sample(
                kuleshov_success, min(method_sample_size, len(kuleshov_success))
            )
        ),
        "ffpbs_overall": set(
            rng.sample(ff_success, min(method_sample_size, len(ff_success)))
        ),
        "ffpbs_unique": set(
            rng.sample(ff_unique, min(unique_sample_size, len(ff_unique)))
        ),
    }

    units = {}
    for cohort, indices in selected.items():
        method = "DT4LM-Kuleshov" if cohort == "kuleshov_overall" else "FF-PBS"
        source = kuleshov if method == "DT4LM-Kuleshov" else ffpbs
        for dataset_index in indices:
            unit = units.setdefault(
                (method, dataset_index),
                {
                    "method": method,
                    "dataset_index": dataset_index,
                    "source": source[dataset_index],
                    "cohorts": [],
                },
            )
            unit["cohorts"].append(cohort)

    ordered_units = list(units.values())
    rng.shuffle(ordered_units)
    review_rows = []
    key_rows = []
    for ordinal, unit in enumerate(ordered_units, start=1):
        review_id = f"human-{ordinal:04d}"
        source = unit["source"]
        review_rows.append(
            {
                "review_id": review_id,
                "original_input": source["original_input"],
                "ground_truth_output": source["ground_truth_output"],
                "candidate_input": source["candidate_input"],
                "reviewer_1_label_preserved": None,
                "reviewer_2_label_preserved": None,
                "final_label_preserved": None,
                "reviewer_1_semantic_preserved": None,
                "reviewer_2_semantic_preserved": None,
                "final_semantic_preserved": None,
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "dataset_index": unit["dataset_index"],
                "method": unit["method"],
                "cohorts": sorted(unit["cohorts"]),
            }
        )
    return review_rows, {
        "schema_version": 2,
        "seed": seed,
        "review_unit_count": len(review_rows),
        "population_counts": {
            "kuleshov_overall": len(kuleshov_success),
            "ffpbs_overall": len(ff_success),
            "ffpbs_unique": len(ff_unique),
        },
        "sample_counts": {name: len(values) for name, values in selected.items()},
        "rows": key_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kuleshov-results", required=True)
    parser.add_argument("--ffpbs-results", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key-output", required=True)
    parser.add_argument("--method-sample-size", type=int, default=100)
    parser.add_argument("--unique-sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=765)
    args = parser.parse_args()

    kuleshov = _index_results(args.kuleshov_results)
    ffpbs = _index_results(args.ffpbs_results)
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = [int(value) for value in manifest["selected_indices"]]
    if list(kuleshov) != expected or list(ffpbs) != expected:
        raise ValueError("Result order must exactly match the supplied manifest.")

    reviews, key = build_sample(
        kuleshov,
        ffpbs,
        method_sample_size=args.method_sample_size,
        unique_sample_size=args.unique_sample_size,
        seed=args.seed,
    )
    key["manifest_sample_count"] = manifest["effective_sample_size"]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in reviews:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    os.makedirs(os.path.dirname(args.key_output) or ".", exist_ok=True)
    with open(args.key_output, "w", encoding="utf-8") as handle:
        json.dump(key, handle, ensure_ascii=True, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
