"""One greedy-search state machine shared by all differential objectives."""

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods.search_method import SearchMethod

from .differential_comparators import ResultComparator, ScalarComparator


class ComparatorGreedySearch(SearchMethod):
    """Greedily advance using an injected result comparator.

    Dynamic and static objectives use ``ScalarComparator`` while LexiDT uses
    ``LexicographicComparator``. Keeping this control flow shared prevents
    search implementation differences from confounding objective comparisons.
    """

    def __init__(self, comparator: ResultComparator = None):
        self.comparator = comparator or ScalarComparator()

    def perform_search(self, initial_result):
        current_text = initial_result.attacked_text
        best_result = initial_result
        search_round = 0

        while best_result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
            transformations = self.get_transformations(
                current_text, original_text=initial_result.attacked_text
            )
            if not transformations:
                return best_result

            # Search metadata is attached before GoalFunction applies its query
            # budget, allowing the observer to record only evaluated candidates.
            for candidate_order, candidate in enumerate(transformations):
                candidate.attack_attrs["search_round"] = search_round
                candidate.attack_attrs["candidate_order"] = candidate_order

            results, search_over = self.get_goal_results(transformations)
            if not results:
                return best_result

            best_result = self.comparator.select(results)
            if (
                best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED
                or search_over
            ):
                return best_result

            current_text = best_result.attacked_text
            search_round += 1

        return best_result

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["comparator"]
