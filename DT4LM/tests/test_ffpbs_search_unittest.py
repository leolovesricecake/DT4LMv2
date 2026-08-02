"""Lightweight FF-PBS state-machine tests without torch imports."""

from collections import OrderedDict
import importlib.util
from pathlib import Path
import sys
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _namespace(name):
    """Install one lightweight package namespace for direct source loading."""

    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return module


def _load(name, relative_path):
    """Load a repository module without importing TextAttack's root package."""

    parts = name.split(".")
    for index in range(1, len(parts)):
        _namespace(".".join(parts[:index]))
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class GoalFunctionResultStatus:
    SUCCEEDED = 0
    SEARCHING = 1
    SKIPPED = 3


goal_results = _namespace("textattack.goal_function_results")
goal_results.GoalFunctionResultStatus = GoalFunctionResultStatus


class SearchMethod:
    """Minimal base matching the methods exercised by the state machine."""

    pass


search_method_module = _namespace("textattack.search_methods.search_method")
search_method_module.SearchMethod = SearchMethod
_load(
    "textattack.search_methods.differential_frontier",
    "textattack/search_methods/differential_frontier.py",
)
search_module = _load(
    "textattack.search_methods.async_differential_beam_search",
    "textattack/search_methods/async_differential_beam_search.py",
)


class FakeAttackedText:
    """Path-aware text object exposing the subset used by FF-PBS."""

    def __init__(self, text, modified_indices=()):
        self.text_input = OrderedDict((("text", text),))
        self.attack_attrs = {
            "modified_indices": set(modified_indices),
            "original_index_map": [0, 1],
            "dataset_index": 7,
        }

    def modification_rate(self, original_text):
        del original_text
        return len(self.attack_attrs["modified_indices"]) / 2


class FakeResult:
    """Constructible goal result compatible with query-cache materialization."""

    def __init__(
        self,
        attacked_text,
        raw_output,
        output,
        goal_status,
        score,
        num_queries,
        ground_truth_output,
    ):
        self.attacked_text = attacked_text
        self.raw_output = raw_output
        self.output = output
        self.goal_status = goal_status
        self.score = score
        self.num_queries = num_queries
        self.ground_truth_output = ground_truth_output
        self.objective_name = "dynamic"


class FakeGoalFunction:
    def __init__(self, query_budget=20):
        self.query_budget = query_budget
        self.num_queries = 1


def _result(text, *, old_margin, new_margin, queries=1, success=False):
    """Build a fake differential result with role-specific search margins."""

    result = FakeResult(
        text,
        raw_output=[0.8, 0.2],
        output=(1 if success else 0),
        goal_status=(
            GoalFunctionResultStatus.SUCCEEDED
            if success
            else GoalFunctionResultStatus.SEARCHING
        ),
        score=(1e6 if success else new_margin),
        num_queries=queries,
        ground_truth_output=0,
    )
    result.new_model_output = {
        "predicted_label": 1 if success else 0,
        "objective_margin": new_margin,
    }
    result.old_model_output = {
        "predicted_label": 0 if old_margin >= 0 else 1,
        "objective_margin": old_margin,
    }
    return result


class SearchHarness:
    """Bind deterministic transformations and model outputs to one search."""

    def __init__(
        self,
        transformations,
        margins,
        beam_size=2,
        ranking="feasibility_pareto",
        infeasible_state_policy="fill",
    ):
        self.goal = FakeGoalFunction()
        self.transformations = transformations
        self.margins = margins
        self.search = search_module.AsyncDifferentialBeamSearch(
            ranking=ranking,
            beam_size=beam_size,
            infeasible_state_policy=infeasible_state_policy,
        )
        self.search.goal_function = self.goal
        self.search.get_transformations = self.get_transformations
        self.search.get_goal_results = self.get_goal_results

    def get_transformations(self, attacked_text, original_text=None):
        del original_text
        return list(self.transformations.get(attacked_text.text_input["text"], ()))

    def get_goal_results(self, candidates):
        remaining = self.goal.query_budget - self.goal.num_queries
        candidates = candidates[:remaining]
        self.goal.num_queries += len(candidates)
        results = []
        for candidate in candidates:
            old_margin, new_margin, success = self.margins[
                candidate.text_input["text"]
            ]
            results.append(
                _result(
                    candidate,
                    old_margin=old_margin,
                    new_margin=new_margin,
                    queries=self.goal.num_queries,
                    success=success,
                )
            )
        return results, self.goal.num_queries == self.goal.query_budget


class AsyncSearchTests(unittest.TestCase):
    def test_transformation_cache_key_preserves_path_state(self):
        first_path = FakeAttackedText("same", modified_indices=(0,))
        second_path = FakeAttackedText("same", modified_indices=(1,))

        self.assertEqual(
            search_module.AsyncDifferentialBeamSearch._query_key(first_path),
            search_module.AsyncDifferentialBeamSearch._query_key(second_path),
        )
        self.assertNotEqual(
            search_module.AsyncDifferentialBeamSearch.transformation_cache_key(
                first_path, {}
            ),
            search_module.AsyncDifferentialBeamSearch.transformation_cache_key(
                second_path, {}
            ),
        )

    def test_same_query_key_keeps_distinct_path_states(self):
        root = FakeAttackedText("root")
        first_path = FakeAttackedText("same", modified_indices=(0,))
        second_path = FakeAttackedText("same", modified_indices=(1,))
        harness = SearchHarness(
            {"root": (first_path, second_path)},
            {"same": (0.2, -0.1, False)},
        )

        final = harness.search.perform_search(
            _result(root, old_margin=0.5, new_margin=-0.5)
        )

        diagnostics = final.search_diagnostics
        self.assertEqual(harness.goal.num_queries, 2)
        self.assertEqual(diagnostics["queried_candidate_count"], 1)
        self.assertEqual(diagnostics["query_cache_hit_count"], 1)
        self.assertEqual(diagnostics["candidate_state_count"], 2)
        self.assertEqual(diagnostics["duplicate_state_count"], 0)

    def test_ffpbs_recovers_after_old_model_error(self):
        root = FakeAttackedText("root")
        escape = FakeAttackedText("escape", modified_indices=(0,))
        success = FakeAttackedText("success", modified_indices=(0, 1))
        harness = SearchHarness(
            {"root": (escape,), "escape": (success,)},
            {
                "escape": (-0.2, -0.1, False),
                "success": (0.3, 0.4, True),
            },
            beam_size=1,
        )

        final = harness.search.perform_search(
            _result(root, old_margin=0.5, new_margin=-0.5)
        )

        self.assertEqual(final.goal_status, GoalFunctionResultStatus.SUCCEEDED)
        self.assertTrue(
            final.search_diagnostics[
                "path_has_post_root_old_prediction_error"
            ]
        )
        self.assertEqual(
            final.search_diagnostics[
                "first_post_root_old_prediction_error_depth"
            ],
            1,
        )
        self.assertEqual(
            final.search_diagnostics[
                "first_recovery_depth_after_old_prediction_error"
            ],
            2,
        )
        self.assertEqual(
            final.search_diagnostics["successful_path"]["depth"], [0, 1, 2]
        )
        self.assertEqual(
            final.search_diagnostics["successful_path"]["old_is_correct"],
            [True, False, True],
        )

    def test_hard_pbs_discards_infeasible_recovery_state(self):
        root = FakeAttackedText("root")
        escape = FakeAttackedText("escape", modified_indices=(0,))
        success = FakeAttackedText("success", modified_indices=(0, 1))
        harness = SearchHarness(
            {"root": (escape,), "escape": (success,)},
            {
                "escape": (-0.2, -0.1, False),
                "success": (0.3, 0.4, True),
            },
            beam_size=1,
            infeasible_state_policy="discard",
        )

        final = harness.search.perform_search(
            _result(root, old_margin=0.5, new_margin=-0.5)
        )

        self.assertNotEqual(final.goal_status, GoalFunctionResultStatus.SUCCEEDED)
        self.assertEqual(
            final.search_diagnostics["discarded_infeasible_state_count"], 1
        )
        self.assertEqual(final.search_diagnostics["hard_discard_rate"], 1)

    def test_hard_pbs_expands_an_old_incorrect_root_once(self):
        root = FakeAttackedText("root")
        success = FakeAttackedText("success", modified_indices=(0,))
        harness = SearchHarness(
            {"root": (success,)},
            {"success": (0.3, 0.4, True)},
            beam_size=1,
            infeasible_state_policy="discard",
        )

        final = harness.search.perform_search(
            _result(root, old_margin=-0.5, new_margin=-0.5)
        )

        self.assertEqual(final.goal_status, GoalFunctionResultStatus.SUCCEEDED)
        self.assertFalse(
            final.search_diagnostics[
                "path_has_post_root_old_prediction_error"
            ]
        )

    def test_infeasible_fill_and_normalized_diversity_are_recorded(self):
        root = FakeAttackedText("root")
        feasible = FakeAttackedText("feasible", modified_indices=(0,))
        infeasible = FakeAttackedText("infeasible", modified_indices=(1,))
        harness = SearchHarness(
            {"root": (feasible, infeasible)},
            {
                "feasible": (0.2, -0.2, False),
                "infeasible": (-0.1, -0.1, False),
            },
            beam_size=2,
        )

        final = harness.search.perform_search(
            _result(root, old_margin=0.5, new_margin=-0.5)
        )

        diagnostics = final.search_diagnostics
        self.assertEqual(diagnostics["frontier_update_count"], 2)
        self.assertEqual(diagnostics["frontier_state_slot_count"], 3)
        self.assertEqual(diagnostics["infeasible_fill_event_rate"], 1)
        self.assertAlmostEqual(
            diagnostics["infeasible_retained_state_rate"], 2 / 3
        )
        self.assertEqual(
            diagnostics["frontier_modified_set_diversity_mean"], 1
        )

    def test_non_top1_first_branch_is_attributed_to_success(self):
        root = FakeAttackedText("root")
        dynamic_top = FakeAttackedText("dynamic-top", modified_indices=(0, 1))
        alternate = FakeAttackedText("alternate", modified_indices=(0,))
        success = FakeAttackedText("success", modified_indices=(0, 1))
        harness = SearchHarness(
            {"root": (dynamic_top, alternate), "alternate": (success,)},
            {
                "dynamic-top": (0.2, 0.9, False),
                "alternate": (0.2, 0.8, False),
                "success": (0.3, 1.0, True),
            },
            beam_size=2,
        )

        final = harness.search.perform_search(
            _result(root, old_margin=0.5, new_margin=-0.5)
        )

        self.assertEqual(final.goal_status, GoalFunctionResultStatus.SUCCEEDED)
        self.assertEqual(final.search_diagnostics["root_dynamic_rank"], 2)


if __name__ == "__main__":
    unittest.main()
