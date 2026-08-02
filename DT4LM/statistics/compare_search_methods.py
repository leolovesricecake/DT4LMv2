#!/usr/bin/env python
"""Paired statistical comparison of two DT4LM search-method runs."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
from statistics import mean, median

import yaml


VALID_STATUSES = frozenset(("successful", "failed", "skipped"))


def _method_identity(config):
    """Return the semantic method name and explicit infeasible policy."""

    configured = config["experiment"]["method"]
    search = config["attack"].get("search") or {}
    epsilon = search.get("epsilon") or {}
    policy = epsilon.get("infeasible_state_policy")
    if policy is None and search.get("ranking") == "epsilon_pareto":
        policy = "feasibility_first"
    method = configured
    if (
        configured == "strict-pbs"
        and epsilon.get("mode") == "strict"
        and "infeasible_state_policy" not in epsilon
    ):
        method = "feasibility-first-pbs"
    return method, policy


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_yaml(path):
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping at {path}.")
    return payload


def _load_run(path):
    """Load and index one self-contained run directory."""

    path = Path(path).resolve()
    manifest = _read_json(path / "sample_manifest.json")
    config = _read_yaml(path / "config.resolved.yaml")
    records = _read_jsonl(path / "results.jsonl")
    indexed = {}
    for row in records:
        status = row.get("result_status")
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid result status {status!r} in {path}.")
        dataset_index = int(row["dataset_index"])
        if dataset_index in indexed:
            raise ValueError(f"Duplicate dataset index {dataset_index} in {path}.")
        indexed[dataset_index] = row
    expected = [int(value) for value in manifest["selected_indices"]]
    if list(indexed) != expected:
        raise ValueError(f"Result order does not match the manifest in {path}.")
    return {
        "path": path,
        "manifest": manifest,
        "config": config,
        "records": indexed,
    }


def _validate_pair(baseline, candidate):
    """Require identical samples, model pair, dataset, and query budget."""

    left_manifest = baseline["manifest"]
    right_manifest = candidate["manifest"]
    for key in ("dataset_id", "split", "selection_sha256", "selected_indices"):
        if left_manifest.get(key) != right_manifest.get(key):
            raise ValueError(f"Run manifests differ at {key!r}.")
    for path in (("dataset", "id"), ("models", "id"), ("attack", "query_budget")):
        left = baseline["config"]
        right = candidate["config"]
        for part in path:
            left = left[part]
            right = right[part]
        if left != right:
            raise ValueError(f"Run configurations differ at {'.'.join(path)}.")


def _quantile(values, probability):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def _paired_bootstrap(left, right, *, samples, seed, statistic=mean):
    """Bootstrap a paired candidate-minus-baseline statistic."""

    if len(left) != len(right):
        raise ValueError("Paired bootstrap inputs must have equal length.")
    if not left:
        return {"estimate": None, "ci95": [None, None], "sample_count": 0}
    differences = [float(r) - float(l) for l, r in zip(left, right)]
    generator = random.Random(seed)
    replicates = []
    for _ in range(samples):
        draw = [differences[generator.randrange(len(differences))] for _ in differences]
        replicates.append(float(statistic(draw)))
    return {
        "estimate": float(statistic(differences)),
        "ci95": [_quantile(replicates, 0.025), _quantile(replicates, 0.975)],
        "sample_count": len(differences),
    }


def _mcnemar_exact(baseline_only, candidate_only):
    """Return the two-sided exact McNemar p-value for discordant pairs."""

    discordant = int(baseline_only) + int(candidate_only)
    if discordant == 0:
        return 1.0
    smaller = min(int(baseline_only), int(candidate_only))
    tail = sum(math.comb(discordant, value) for value in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _wilcoxon(left, right):
    """Run Wilcoxon lazily so artifact validation does not require SciPy."""

    if not left:
        return {"statistic": None, "pvalue": None, "status": "no_samples"}
    if all(float(l) == float(r) for l, r in zip(left, right)):
        return {"statistic": 0.0, "pvalue": 1.0, "status": "all_equal"}
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(left, right, alternative="two-sided")
        return {
            "statistic": float(result.statistic),
            "pvalue": float(result.pvalue),
            "status": "completed",
        }
    except Exception as exc:
        return {
            "statistic": None,
            "pvalue": None,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def compare_runs(baseline, candidate, *, bootstrap_samples=10000, seed=765):
    """Build paired outcome, efficiency, quality, and mechanism statistics."""

    baseline = _load_run(baseline)
    candidate = _load_run(candidate)
    _validate_pair(baseline, candidate)
    indices = list(baseline["records"])
    query_budget = int(baseline["config"]["attack"]["query_budget"])

    contingency = Counter()
    baseline_success = []
    candidate_success = []
    common_baseline_queries = []
    common_candidate_queries = []
    common_baseline_modifications = []
    common_candidate_modifications = []
    penalized_baseline = []
    penalized_candidate = []
    candidate_unique_rows = []

    for dataset_index in indices:
        left = baseline["records"][dataset_index]
        right = candidate["records"][dataset_index]
        left_status = left["result_status"]
        right_status = right["result_status"]
        contingency[(left_status, right_status)] += 1
        if (left_status == "skipped") != (right_status == "skipped"):
            raise ValueError(
                f"Skip status differs for dataset index {dataset_index}; "
                "runs are not paired."
            )
        if left_status == "skipped":
            continue

        left_succeeded = left_status == "successful"
        right_succeeded = right_status == "successful"
        baseline_success.append(int(left_succeeded))
        candidate_success.append(int(right_succeeded))
        penalized_baseline.append(
            int(left["queries_to_success"]) if left_succeeded else query_budget
        )
        penalized_candidate.append(
            int(right["queries_to_success"]) if right_succeeded else query_budget
        )
        if left_succeeded and right_succeeded:
            common_baseline_queries.append(int(left["queries_to_success"]))
            common_candidate_queries.append(int(right["queries_to_success"]))
            common_baseline_modifications.append(float(left["modification_rate"]))
            common_candidate_modifications.append(float(right["modification_rate"]))
        elif right_succeeded and not left_succeeded:
            candidate_unique_rows.append(right)

    baseline_only = sum(
        left == 1 and right == 0
        for left, right in zip(baseline_success, candidate_success)
    )
    candidate_only = sum(
        left == 0 and right == 1
        for left, right in zip(baseline_success, candidate_success)
    )
    unique_non_top1 = [
        row["search_diagnostics"].get("root_dynamic_rank") > 1
        for row in candidate_unique_rows
        if isinstance(row.get("search_diagnostics"), dict)
        and row["search_diagnostics"].get("root_dynamic_rank") is not None
    ]
    unique_escape = [
        bool(row["search_diagnostics"].get("path_has_negative_old_margin"))
        for row in candidate_unique_rows
        if isinstance(row.get("search_diagnostics"), dict)
        and row["search_diagnostics"].get("path_has_negative_old_margin") is not None
    ]
    unique_old_prediction_error = [
        bool(row["search_diagnostics"].get("path_has_old_prediction_error"))
        for row in candidate_unique_rows
        if isinstance(row.get("search_diagnostics"), dict)
        and row["search_diagnostics"].get("path_has_old_prediction_error")
        is not None
    ]
    unique_post_root_escape = [
        bool(
            row["search_diagnostics"].get(
                "path_has_post_root_negative_old_margin"
            )
        )
        for row in candidate_unique_rows
        if isinstance(row.get("search_diagnostics"), dict)
        and row["search_diagnostics"].get(
            "path_has_post_root_negative_old_margin"
        )
        is not None
    ]

    success_bootstrap = _paired_bootstrap(
        baseline_success,
        candidate_success,
        samples=bootstrap_samples,
        seed=seed,
    )
    common_query_bootstrap = _paired_bootstrap(
        common_baseline_queries,
        common_candidate_queries,
        samples=bootstrap_samples,
        seed=seed + 1,
        statistic=median,
    )
    common_modification_bootstrap = _paired_bootstrap(
        common_baseline_modifications,
        common_candidate_modifications,
        samples=bootstrap_samples,
        seed=seed + 2,
        statistic=median,
    )
    penalized_bootstrap = _paired_bootstrap(
        penalized_baseline,
        penalized_candidate,
        samples=bootstrap_samples,
        seed=seed + 3,
        statistic=median,
    )

    baseline_method, baseline_policy = _method_identity(baseline["config"])
    candidate_method, candidate_policy = _method_identity(candidate["config"])
    return {
        "schema_version": 1,
        "baseline_run": str(baseline["path"]),
        "candidate_run": str(candidate["path"]),
        "dataset": baseline["config"]["dataset"]["id"],
        "model_pair": baseline["config"]["models"]["id"],
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "baseline_infeasible_state_policy": baseline_policy,
        "candidate_infeasible_state_policy": candidate_policy,
        "baseline_attack_seed": baseline["config"]["experiment"].get("seed"),
        "candidate_attack_seed": candidate["config"]["experiment"].get("seed"),
        "query_budget": query_budget,
        "manifest_selection_sha256": baseline["manifest"].get("selection_sha256"),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "status_contingency": {
            f"baseline_{left}__candidate_{right}": contingency[(left, right)]
            for left in sorted(VALID_STATUSES)
            for right in sorted(VALID_STATUSES)
        },
        "attackable_sample_count": len(baseline_success),
        "baseline_only_success": baseline_only,
        "candidate_only_success": candidate_only,
        "common_success": sum(
            left == right == 1
            for left, right in zip(baseline_success, candidate_success)
        ),
        "common_failure": sum(
            left == right == 0
            for left, right in zip(baseline_success, candidate_success)
        ),
        "success_rate_difference": success_bootstrap,
        "mcnemar_exact_pvalue": _mcnemar_exact(baseline_only, candidate_only),
        "common_success_queries": {
            "paired_median_difference": common_query_bootstrap,
            "wilcoxon": _wilcoxon(
                common_baseline_queries, common_candidate_queries
            ),
        },
        "common_success_modification_rate": {
            "paired_median_difference": common_modification_bootstrap,
            "wilcoxon": _wilcoxon(
                common_baseline_modifications, common_candidate_modifications
            ),
        },
        "budget_penalized_queries": {
            "paired_median_difference": penalized_bootstrap,
            "paired_mean_difference": _paired_bootstrap(
                penalized_baseline,
                penalized_candidate,
                samples=bootstrap_samples,
                seed=seed + 4,
            ),
            "wilcoxon": _wilcoxon(penalized_baseline, penalized_candidate),
        },
        "candidate_unique_success_mechanism": {
            "sample_count": len(candidate_unique_rows),
            "non_top1_rate": mean(unique_non_top1) if unique_non_top1 else None,
            "escape_path_rate": mean(unique_escape) if unique_escape else None,
            "old_prediction_error_path_rate": (
                mean(unique_old_prediction_error)
                if unique_old_prediction_error
                else None
            ),
            "post_root_escape_path_rate": (
                mean(unique_post_root_escape)
                if unique_post_root_escape
                else None
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline run directory.")
    parser.add_argument("--candidate", required=True, help="Candidate run directory.")
    parser.add_argument("--o", default="paired_comparison.json")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=765)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")
    output = Path(args.o).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = compare_runs(
        args.baseline,
        args.candidate,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
    temporary.replace(output)


if __name__ == "__main__":
    main()
