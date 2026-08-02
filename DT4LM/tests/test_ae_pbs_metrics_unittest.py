"""Standard-library regression tests for AE-PBS metric denominators."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dt4lm_evaluate_improvements",
    ROOT / "statistics/evaluate_improvements.py",
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class MetricTests(unittest.TestCase):
    def test_core_metrics_keep_three_state_and_search_denominators(self):
        manifest = {
            "effective_sample_size": 3,
            "selected_indices": [3, 5, 8],
        }
        records = [
            {
                "dataset_index": 3,
                "result_status": "successful",
                "initial_state": "both_correct",
                "model_pair_queries": 10,
                "queries_to_success": 10,
                "modification_rate": 0.2,
                "search_diagnostics": {
                    "epsilon_mode": "adaptive",
                    "expansion_count": 2,
                    "max_depth": 2,
                    "constraint_passed_candidate_count": 4,
                    "duplicate_state_count": 1,
                    "query_cache_hit_count": 1,
                    "query_cache_miss_count": 3,
                    "budget_truncated_candidate_count": 0,
                    "root_dynamic_rank": 2,
                    "path_has_negative_old_margin": True,
                    "path_has_old_prediction_error": False,
                    "epsilon_zero_initialization": False,
                    "epsilon_to_root_margin_ratio": 0.5,
                    "epsilon_initialization_expansion": 2,
                },
            },
            {
                "dataset_index": 5,
                "result_status": "failed",
                "initial_state": "both_wrong",
                "model_pair_queries": 1000,
                "queries_to_success": None,
                "modification_rate": 0.0,
                "search_diagnostics": {
                    "epsilon_mode": "adaptive",
                    "expansion_count": 3,
                    "max_depth": 1,
                    "constraint_passed_candidate_count": 2,
                    "duplicate_state_count": 0,
                    "query_cache_hit_count": 0,
                    "query_cache_miss_count": 2,
                    "budget_truncated_candidate_count": 1,
                    "epsilon_zero_initialization": True,
                    "epsilon_to_root_margin_ratio": 0.0,
                    "epsilon_initialization_expansion": 2,
                },
            },
            {
                "dataset_index": 8,
                "result_status": "skipped",
                "initial_state": "already_differential",
                "model_pair_queries": 1,
                "queries_to_success": None,
                "modification_rate": 0.0,
            },
        ]

        core, queries = evaluation.core_metrics(
            records,
            manifest,
            success_budgets=[100, 500, 1000],
            query_budget=1000,
        )

        self.assertEqual(core["model_pair_qps"], 1011)
        self.assertEqual(core["budget_penalized_query_count"], 2)
        self.assertEqual(core["budget_penalized_queries_mean"], 505)
        self.assertAlmostEqual(core["duplicate_state_rate"], 1 / 6)
        self.assertEqual(core["non_top1_path_rate"], 1)
        self.assertEqual(core["epsilon_zero_initialization_rate"], 0.5)
        self.assertEqual(
            queries["data"],
            {"dataset_index": [3], "queries_to_success": [10]},
        )


if __name__ == "__main__":
    unittest.main()
