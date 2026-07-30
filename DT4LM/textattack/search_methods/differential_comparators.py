"""Comparison policies for the shared differential greedy search."""

from abc import ABC, abstractmethod
from typing import Sequence

from textattack.goal_function_results import GoalFunctionResultStatus

from textattack.goal_functions.classification.differential_objectives import (
    LexiScore,
)


class ResultComparator(ABC):
    """Select one result without owning any search-control behavior."""

    @abstractmethod
    def select(self, results: Sequence):
        raise NotImplementedError()


class ScalarComparator(ResultComparator):
    """Select the first maximum scalar score, matching NumPy ``argmax``."""

    def select(self, results: Sequence):
        if not results:
            raise ValueError("Cannot select from an empty result sequence.")
        successful = [
            result
            for result in results
            if result.goal_status == GoalFunctionResultStatus.SUCCEEDED
        ]
        selectable = successful or results
        # Python's max is stable and therefore preserves the old first-max tie
        # behavior. Restricting to current successes is inert for Dynamic's
        # 1e6 score and prevents Static from walking past a valid regression.
        return max(
            enumerate(selectable), key=lambda item: item[1].score
        )[1]


class LexicographicComparator(ResultComparator):
    """Select LexiDT tuples and prefer minimum cost among current successes."""

    def select(self, results: Sequence):
        if not results:
            raise ValueError("Cannot select from an empty result sequence.")
        successful = [
            result
            for result in results
            if result.goal_status == GoalFunctionResultStatus.SUCCEEDED
        ]
        if successful:
            # negative_cost is larger for cheaper candidates. ``max`` keeps
            # stable ordering when two successful candidates have equal cost.
            return max(successful, key=lambda result: result.score.negative_cost)

        for result in results:
            if not isinstance(result.score, LexiScore):
                raise TypeError(
                    "LexicographicComparator requires every result score to "
                    "be a LexiScore."
                )
        return max(results, key=lambda result: result.score.as_tuple())


def comparator_for_objective(objective_name: str) -> ResultComparator:
    """Map objective names to the only comparison policy they may use."""

    if objective_name == "lexi":
        return LexicographicComparator()
    if objective_name in {"dynamic", "static"}:
        return ScalarComparator()
    raise ValueError(f"No comparator is registered for {objective_name!r}.")
