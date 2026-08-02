"""Standard-library regression tests for FF-PBS metric definitions."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dt4lm_evaluate_improvements",
    ROOT / "statistics/evaluate_improvements.py",
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
AGGREGATE_SPEC = importlib.util.spec_from_file_location(
    "dt4lm_aggregate_improvements",
    ROOT / "statistics/aggregate_improvements.py",
)
aggregate = importlib.util.module_from_spec(AGGREGATE_SPEC)
AGGREGATE_SPEC.loader.exec_module(aggregate)
SAMPLE_SPEC = importlib.util.spec_from_file_location(
    "dt4lm_sample_human_evaluation",
    ROOT / "statistics/sample_human_evaluation.py",
)
human_sample = importlib.util.module_from_spec(SAMPLE_SPEC)
SAMPLE_SPEC.loader.exec_module(human_sample)
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "dt4lm_analyze_human_evaluation",
    ROOT / "statistics/analyze_human_evaluation.py",
)
human_analysis = importlib.util.module_from_spec(ANALYSIS_SPEC)
ANALYSIS_SPEC.loader.exec_module(human_analysis)


class MetricTests(unittest.TestCase):
    @staticmethod
    def _write_json(path, payload):
        """Write one synthetic artifact used by integration-level tests."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_core_metrics_follow_paper_denominators_and_query_schema(self):
        manifest = {
            "effective_sample_size": 3,
            "selected_indices": [3, 5, 8],
        }
        records = [
            {
                "schema_version": 4,
                "dataset_index": 3,
                "result_status": "successful",
                "initial_state": "both_correct",
                "model_pair_queries": 10,
                "queries_to_success": 10,
                "modification_rate": 0.2,
                "search_diagnostics": {
                    "ranking": "feasibility_pareto",
                    "infeasible_state_policy": "fill",
                    "expansion_count": 2,
                    "max_depth": 2,
                    "constraint_passed_candidate_count": 4,
                    "duplicate_state_count": 1,
                    "query_cache_hit_count": 1,
                    "query_cache_miss_count": 3,
                    "budget_truncated_candidate_count": 0,
                    "candidate_state_count": 4,
                    "frontier_update_count": 2,
                    "frontier_state_slot_count": 6,
                    "rank1_state_count": 3,
                    "modified_diversity_ratio_sum": 1.5,
                    "depth_diversity_ratio_sum": 1.0,
                    "infeasible_fill_event_count": 1,
                    "infeasible_retained_state_count": 2,
                    "frontier_size_max": 3,
                    "frontier_sort_seconds": 0.02,
                    "root_dynamic_rank": 2,
                    "success_path_depth": 2,
                    "path_has_post_root_old_prediction_error": True,
                    "first_post_root_old_prediction_error_depth": 1,
                    "first_recovery_depth_after_old_prediction_error": 2,
                },
            },
            {
                "schema_version": 4,
                "dataset_index": 5,
                "result_status": "failed",
                "initial_state": "both_wrong",
                "model_pair_queries": 1000,
                "queries_to_success": None,
                "modification_rate": 0.0,
                "search_diagnostics": {
                    "ranking": "feasibility_pareto",
                    "infeasible_state_policy": "fill",
                    "expansion_count": 3,
                    "max_depth": 1,
                    "constraint_passed_candidate_count": 2,
                    "duplicate_state_count": 0,
                    "query_cache_hit_count": 0,
                    "query_cache_miss_count": 2,
                    "budget_truncated_candidate_count": 1,
                    "candidate_state_count": 2,
                    "frontier_update_count": 1,
                    "frontier_state_slot_count": 2,
                    "rank1_state_count": 1,
                    "modified_diversity_ratio_sum": 0.5,
                    "depth_diversity_ratio_sum": 0.5,
                    "infeasible_fill_event_count": 1,
                    "infeasible_retained_state_count": 1,
                    "frontier_size_max": 2,
                    "frontier_sort_seconds": 0.01,
                },
            },
            {
                "schema_version": 4,
                "dataset_index": 8,
                "result_status": "skipped",
                "initial_state": "already_differential",
                "model_pair_queries": 1,
                "queries_to_success": None,
                "modification_rate": 0.0,
            },
        ]

        core, query_data = evaluation.core_metrics(
            records,
            manifest,
            success_budgets=[100, 500, 1000],
            query_budget=1000,
        )

        self.assertEqual(core["schema_version"], 4)
        self.assertEqual(core["model_pair_qps"], 1011)
        self.assertEqual(core["bpqc"], 505)
        self.assertEqual(core["normalized_bpqc"], 0.505)
        self.assertEqual(core["success_at_100"], 0.5)
        self.assertAlmostEqual(core["success_query_auc"], 991 / 2000)
        self.assertAlmostEqual(core["duplicate_state_rate"], 1 / 6)
        self.assertEqual(core["non_top1_path_rate"], 1)
        self.assertEqual(core["post_root_old_prediction_error_path_rate"], 1)
        self.assertEqual(core["recover_first_infeasible_depth_mean"], 1)
        self.assertEqual(core["recover_first_recovery_depth_mean"], 2)
        self.assertEqual(core["recover_depth_span_mean"], 1)
        self.assertAlmostEqual(core["infeasible_fill_event_rate"], 2 / 3)
        self.assertAlmostEqual(core["infeasible_retained_state_rate"], 3 / 8)
        self.assertAlmostEqual(core["frontier_size_mean"], 8 / 3)
        self.assertAlmostEqual(
            core["frontier_modified_set_diversity_mean"], 2 / 3
        )
        self.assertEqual(
            query_data["data"],
            {
                "dataset_index": [3, 5, 8],
                "result_status": ["successful", "failed", "skipped"],
                "model_pair_queries": [10, 1000, 1],
                "queries_to_success": [10, None, None],
                "budget_penalized_queries": [10, 1000, None],
            },
        )

    def test_hard_discard_rate_uses_all_evaluated_post_root_states(self):
        diagnostics = [
            {
                "result_status": "failed",
                "search_diagnostics": {
                    "ranking": "feasibility_pareto",
                    "infeasible_state_policy": "discard",
                    "candidate_state_count": 8,
                    "discarded_infeasible_state_count": 3,
                },
            }
        ]

        result = evaluation._search_diagnostic_metrics(diagnostics, ["failed"])

        self.assertEqual(result["hard_discarded_infeasible_state_count"], 3)
        self.assertEqual(result["hard_discard_rate"], 3 / 8)

    def test_human_protocol_samples_methods_and_unique_success_separately(self):
        def row(index, status):
            return {
                "dataset_index": index,
                "result_status": status,
                "original_input": {"sentence": f"original-{index}"},
                "candidate_input": {"sentence": f"candidate-{index}"},
                "ground_truth_output": 0,
            }

        base = {
            0: row(0, "successful"),
            1: row(1, "successful"),
            2: row(2, "failed"),
        }
        ffpbs = {
            0: row(0, "failed"),
            1: row(1, "successful"),
            2: row(2, "successful"),
        }

        reviews, key = human_sample.build_sample(
            base,
            ffpbs,
            method_sample_size=2,
            unique_sample_size=1,
            seed=7,
        )
        rates = human_analysis._rates(
            [
                {"label_preserved": True, "semantic_preserved": True},
                {"label_preserved": True, "semantic_preserved": False},
                {"label_preserved": False, "semantic_preserved": True},
            ]
        )

        self.assertEqual(len(reviews), 4)
        self.assertEqual(key["population_counts"]["ffpbs_unique"], 1)
        self.assertEqual(rates["lpr"], 2 / 3)
        self.assertEqual(rates["spr"], 2 / 3)
        self.assertEqual(rates["hvr"], 1 / 3)

    def test_schema_v4_artifacts_aggregate_into_paper_facing_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_dir = Path(temporary)
            run_dir = input_dir / "sst2" / "pair" / "ff-pbs"
            run_dir.mkdir(parents=True)
            config = {
                "experiment": {
                    "id": "sst2-pair-ff-pbs",
                    "method": "ff-pbs",
                    "seed": 765,
                },
                "dataset": {"id": "sst2", "evaluation": {"split": "test"}},
                "models": {
                    "id": "pair",
                    "old": {"name_or_path": "old", "revision": None},
                    "new": {"name_or_path": "new", "revision": None},
                },
                "attack": {
                    "recipe": "kuleshov_var",
                    "differential_objective": "dynamic",
                    "query_budget": 1000,
                    "search": {
                        "method": "async_frontier",
                        "ranking": "feasibility_pareto",
                        "beam_size": 5,
                        "infeasible_state_policy": "fill",
                    },
                },
            }
            (run_dir / "config.resolved.yaml").write_text(
                yaml.safe_dump(config), encoding="utf-8"
            )
            self._write_json(
                run_dir / "sample_manifest.json",
                {
                    "dataset_id": "sst2",
                    "split": "test",
                    "effective_sample_size": 2,
                    "selected_indices": [4, 9],
                    "seed": 765,
                    "selection_sha256": "selection",
                },
            )
            self._write_json(
                run_dir / "provenance.json", {"created_at": "now"}
            )
            core = {
                "schema_version": 4,
                "total": 2,
                "successful": 1,
                "failed": 1,
                "skipped": 0,
                "attackable": 2,
                "query_budget": 1000,
                "paper_gsr": 0.5,
                "sample_generation_rate": 0.5,
                "model_pair_qps": 1010,
                "success_at_100": 0.5,
                "initial_state_counts": {"both_correct": 2},
                "resources": {
                    "end_to_end_seconds": 3.0,
                    "frontier_sort_seconds": 0.1,
                },
                "evaluation_runtime": {"evaluated_at": "later"},
            }
            self._write_json(run_dir / "metrics" / "core.json", core)
            quality_metrics = {
                name: {
                    "status": "completed",
                    "values": (
                        {"precision": 0.9, "recall": 0.8, "f1": 0.85}
                        if name == "bertscore"
                        else {"value": 0.7}
                    ),
                }
                for name in aggregate.QUALITY_METRICS
            }
            self._write_json(
                run_dir / "metrics" / "quality.json",
                {
                    "schema_version": 4,
                    "successful_sample_count": 1,
                    "metrics": quality_metrics,
                    "evaluation_runtime": {"evaluated_at": "later"},
                },
            )
            self._write_json(
                run_dir / "metrics" / "query_data.json",
                {
                    "schema_version": 4,
                    "sample_count": 2,
                    "successful_sample_count": 1,
                    "query_budget": 1000,
                    "data": {
                        "dataset_index": [4, 9],
                        "result_status": ["successful", "failed"],
                        "model_pair_queries": [10, 1000],
                        "queries_to_success": [10, None],
                        "budget_penalized_queries": [10, 1000],
                    },
                },
            )

            output = input_dir / "summary.csv"
            self.assertEqual(aggregate.write_summary(input_dir, output), 1)
            with open(output, encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["method"], "ff-pbs")
            self.assertEqual(rows[0]["frontier_ranking"], "feasibility_pareto")
            self.assertEqual(rows[0]["paper_gsr"], "0.5")
            self.assertEqual(rows[0]["frontier_sort_seconds"], "0.1")


if __name__ == "__main__":
    unittest.main()
