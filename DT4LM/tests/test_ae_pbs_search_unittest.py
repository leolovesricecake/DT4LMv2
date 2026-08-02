"""Lightweight asynchronous-search tests without torch or pytest imports."""

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
    """Path-aware text object exposing the subset used by AE-PBS."""

    def __init__(self, text, modified_indices=()):
        self.text_input = OrderedDict((("text", text),))
        self.attack_attrs = {
            "modified_indices": set(modified_indices),
            "original_index_map": [0, 1],
            "dataset_index": 7,
        }

    def modification_rate(self, original_text):
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

    def __init__(self, transformations, margins, beam_size=2):
        self.goal = FakeGoalFunction()
        self.transformations = transformations
        self.margins = margins
        self.search = search_module.AsyncDifferentialBeamSearch(
            ranking="epsilon_pareto",
            beam_size=beam_size,
            epsilon_mode="adaptive",
            epsilon_initial_quantile=0.75,
            epsilon_initialization_max_expansions=2,
            epsilon_decay="quadratic",
        )
        self.search.goal_function = self.goal
        self.search.get_transformations = self.get_transformations
        self.search.get_goal_results = self.get_goal_results

    def get_transformations(self, attacked_text, original_text=None):
        del original_text
        text = attacked_text.text_input["text"]
        return list(self.transformations.get(text, ()))

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

    def test_second_expansion_can_initialize_adaptive_epsilon(self):
        root = FakeAttackedText("root")
        level_one = FakeAttackedText("level-one", modified_indices=(0,))
        level_two = FakeAttackedText("level-two", modified_indices=(0, 1))
        harness = SearchHarness(
            {"root": (level_one,), "level-one": (level_two,)},
            {
                "level-one": (0.3, -0.2, False),
                "level-two": (-0.4, 0.1, False),
            },
            beam_size=1,
        )

        final = harness.search.perform_search(
            _result(root, old_margin=0.5, new_margin=-0.5)
        )

        diagnostics = final.search_diagnostics
        self.assertEqual(diagnostics["epsilon_initialization_expansion"], 2)
        self.assertAlmostEqual(diagnostics["epsilon_0"], 0.4)
        self.assertFalse(diagnostics["epsilon_zero_initialization"])

    def test_early_success_freezes_unneeded_epsilon_as_zero(self):
        root = FakeAttackedText("root")
        success = FakeAttackedText("success", modified_indices=(0,))
        harness = SearchHarness(
            {"root": (success,)},
            {"success": (0.3, 0.4, True)},
            beam_size=1,
        )

        final = harness.search.perform_search(
            _result(root, old_margin=0.5, new_margin=-0.5)
        )

        diagnostics = final.search_diagnostics
        self.assertEqual(diagnostics["epsilon_0"], 0.0)
        self.assertTrue(diagnostics["epsilon_zero_initialization"])
        self.assertEqual(diagnostics["epsilon_initialization_expansion"], 1)

    def test_success_can_recover_after_negative_old_margin(self):
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
        self.assertTrue(final.search_diagnostics["path_has_negative_old_margin"])
        self.assertEqual(final.search_diagnostics["root_dynamic_rank"], 1)


if __name__ == "__main__":
    unittest.main()
