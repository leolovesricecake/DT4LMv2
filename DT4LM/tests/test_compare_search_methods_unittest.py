"""Standard-library tests for paired search-method result analysis."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dt4lm_compare_search_methods",
    ROOT / "statistics/compare_search_methods.py",
)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def _write_run(path, rows, method):
    """Write the minimal self-contained artifacts required by comparison."""

    path.mkdir(parents=True)
    manifest = {
        "dataset_id": "sst2",
        "split": "test",
        "selection_sha256": "same-selection",
        "selected_indices": [0, 1, 2],
    }
    config = {
        "experiment": {"method": method},
        "dataset": {"id": "sst2"},
        "models": {"id": "old-new"},
        "attack": {"query_budget": 100},
    }
    (path / "sample_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (path / "config.resolved.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    (path / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class ComparisonTests(unittest.TestCase):
    def test_paired_outcomes_and_penalized_queries(self):
        baseline_rows = [
            {
                "dataset_index": 0,
                "result_status": "successful",
                "queries_to_success": 20,
                "modification_rate": 0.2,
            },
            {
                "dataset_index": 1,
                "result_status": "failed",
                "queries_to_success": None,
                "modification_rate": 0.0,
            },
            {
                "dataset_index": 2,
                "result_status": "skipped",
                "queries_to_success": None,
                "modification_rate": 0.0,
            },
        ]
        candidate_rows = [
            {
                "dataset_index": 0,
                "result_status": "successful",
                "queries_to_success": 10,
                "modification_rate": 0.1,
                "search_diagnostics": {
                    "root_dynamic_rank": 1,
                    "path_has_negative_old_margin": False,
                },
            },
            {
                "dataset_index": 1,
                "result_status": "successful",
                "queries_to_success": 40,
                "modification_rate": 0.3,
                "search_diagnostics": {
                    "root_dynamic_rank": 2,
                    "path_has_negative_old_margin": True,
                    "path_has_old_prediction_error": True,
                    "path_has_post_root_negative_old_margin": True,
                },
            },
            {
                "dataset_index": 2,
                "result_status": "skipped",
                "queries_to_success": None,
                "modification_rate": 0.0,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root / "baseline", baseline_rows, "base")
            _write_run(root / "candidate", candidate_rows, "ae-pbs")
            # SciPy is optional for artifact assembly; this test isolates the
            # deterministic pairing and bootstrap layers.
            original_wilcoxon = comparison._wilcoxon
            comparison._wilcoxon = lambda left, right: {"status": "stubbed"}
            try:
                result = comparison.compare_runs(
                    root / "baseline",
                    root / "candidate",
                    bootstrap_samples=100,
                    seed=7,
                )
            finally:
                comparison._wilcoxon = original_wilcoxon

        self.assertEqual(result["candidate_only_success"], 1)
        self.assertEqual(result["common_success"], 1)
        self.assertEqual(result["attackable_sample_count"], 2)
        self.assertEqual(
            result["candidate_unique_success_mechanism"]["non_top1_rate"], 1
        )
        self.assertEqual(
            result["candidate_unique_success_mechanism"][
                "old_prediction_error_path_rate"
            ],
            1,
        )
        self.assertEqual(
            result["candidate_unique_success_mechanism"][
                "post_root_escape_path_rate"
            ],
            1,
        )
        self.assertEqual(
            result["budget_penalized_queries"]["paired_mean_difference"][
                "estimate"
            ],
            -35,
        )


if __name__ == "__main__":
    unittest.main()
