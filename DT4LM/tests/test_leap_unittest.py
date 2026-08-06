"""Focused LEAP fixed-dimensional state regression tests."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _namespace(name):
    """Install a lightweight namespace for loading LEAP without TextAttack."""

    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return module


class GoalFunctionResultStatus:
    SUCCEEDED = 0
    SEARCHING = 1


class PopulationBasedSearch:
    pass


class PopulationMember:
    def __init__(self, attacked_text, result=None):
        self.attacked_text = attacked_text
        self.result = result

    @property
    def words(self):
        return self.attacked_text.words

    @property
    def score(self):
        return self.result.score


goal_results = _namespace("textattack.goal_function_results")
goal_results.GoalFunctionResultStatus = GoalFunctionResultStatus
search_methods = _namespace("textattack.search_methods")
search_methods.PopulationBasedSearch = PopulationBasedSearch
search_methods.PopulationMember = PopulationMember
shared = _namespace("textattack.shared")
shared.utils = types.SimpleNamespace()
validators = _namespace("textattack.shared.validators")
validators.transformation_consists_of_word_swaps = lambda transformation: True

spec = importlib.util.spec_from_file_location(
    "textattack.search_methods.leap_regression",
    ROOT / "textattack/search_methods/leap.py",
)
leap_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(leap_module)


class FakeText:
    """Minimal attacked text with controllable replacement tokenization."""

    def __init__(self, words, newly_modified=(), replacement_words=None):
        self.words = list(words)
        self.attack_attrs = {
            "newly_modified_indices": set(newly_modified),
            "modified_indices": set(),
        }
        self._replacement_words = replacement_words

    @property
    def num_words(self):
        return len(self.words)

    def replace_words_at_indices(self, indices, new_words):
        del indices, new_words
        replacement = self._replacement_words or self.words
        return FakeText(replacement)


class FakeResult:
    def __init__(self, attacked_text, score=0.0):
        self.attacked_text = attacked_text
        self.score = score


class LeapInvariantTests(unittest.TestCase):
    def test_neighbor_generation_discards_dimension_changing_swaps(self):
        search = leap_module.LEAP(pop_size=2, max_iters=1)
        root = FakeText(["one", "two"])
        invalid = FakeText(["one", "split", "word"], newly_modified=(0, 1))
        valid = FakeText(["one", "second"], newly_modified=(1,))
        queried = []
        search.get_transformations = lambda *args, **kwargs: [invalid, valid]

        def get_goal_results(candidates):
            queried.extend(candidates)
            return [FakeResult(candidate, 1.0) for candidate in candidates], False

        search.get_goal_results = get_goal_results
        neighbors, probabilities = search._get_best_neighbors(
            FakeResult(root), FakeResult(root)
        )

        self.assertEqual(queried, [valid])
        self.assertTrue(all(result.attacked_text.num_words == 2 for result in neighbors))
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0)

    def test_turn_rejects_retokenized_output(self):
        search = leap_module.LEAP(
            pop_size=2,
            max_iters=1,
            post_turn_check=False,
            max_turn_retries=1,
        )
        source = PopulationMember(
            FakeText(["one", "two"], replacement_words=["one", "split", "word"]),
            FakeResult(FakeText(["one", "two"])),
        )
        target = PopulationMember(FakeText(["first", "second"]))

        with mock.patch.object(leap_module.np.random, "uniform", return_value=0.0):
            turned = search._turn(source, target, np.ones(2), source.attacked_text)

        self.assertIs(turned, source)
        self.assertEqual(turned.attacked_text.num_words, 2)


if __name__ == "__main__":
    unittest.main()
