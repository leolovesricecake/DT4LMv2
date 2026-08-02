"""Pure FF-PBS ranking tests without importing the TextAttack package."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dt4lm_differential_frontier",
    ROOT / "textattack/search_methods/differential_frontier.py",
)
frontier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frontier)


def _state(
    state_id,
    *,
    old_margin=0.1,
    new_margin=0.0,
    cost=0.1,
    dynamic_score=0.0,
    generation_order=None,
    old_is_correct=True,
):
    """Build the minimal state protocol consumed by ranking functions."""

    return SimpleNamespace(
        state_id=state_id,
        old_correct_margin=old_margin,
        new_error_margin=new_margin,
        modification_cost=cost,
        dynamic_score=dynamic_score,
        old_is_correct=old_is_correct,
        generation_order=(state_id if generation_order is None else generation_order),
    )


class FrontierTests(unittest.TestCase):
    def test_non_dominated_fronts_preserve_margin_cost_tradeoffs(self):
        cheap = _state(1, new_margin=0.1, cost=0.1)
        aggressive = _state(2, new_margin=0.8, cost=0.4)
        dominated = _state(3, new_margin=0.0, cost=0.5)

        fronts = frontier.non_dominated_fronts([cheap, aggressive, dominated])

        self.assertEqual([state.state_id for state in fronts[0]], [1, 2])
        self.assertEqual([state.state_id for state in fronts[1]], [3])

    def test_prediction_feasibility_precedes_margin_violation(self):
        feasible = _state(
            1, old_margin=-0.1, old_is_correct=True, new_margin=-1.0
        )
        close = _state(
            2, old_margin=-0.2, old_is_correct=False, new_margin=2.0
        )
        far = _state(
            3, old_margin=-0.8, old_is_correct=False, new_margin=3.0
        )

        ordered, metadata = frontier.feasibility_pareto_order(
            [far, close, feasible]
        )

        self.assertEqual([state.state_id for state in ordered], [1, 2, 3])
        self.assertTrue(metadata[1].feasible)
        self.assertAlmostEqual(metadata[2].violation, 0.2)
        self.assertAlmostEqual(metadata[3].violation, 0.8)

    def test_crowding_distance_and_generation_order_are_stable(self):
        states = [
            _state(1, new_margin=0.0, cost=0.1, generation_order=2),
            _state(2, new_margin=0.5, cost=0.2, generation_order=1),
            _state(3, new_margin=1.0, cost=0.3, generation_order=0),
        ]

        selected, metadata = frontier.select_frontier(
            states, beam_size=2, ranking="feasibility_pareto"
        )

        self.assertEqual([state.state_id for state in selected], [3, 1])
        self.assertEqual(metadata[1].crowding_distance, float("inf"))
        self.assertEqual(metadata[3].crowding_distance, float("inf"))

    def test_mnew_ablation_does_not_use_feasible_modification_cost(self):
        earlier_expensive = _state(
            1, new_margin=0.7, cost=0.9, generation_order=1
        )
        later_cheap = _state(2, new_margin=0.7, cost=0.1, generation_order=2)

        ordered, _ = frontier.feasibility_mnew_order(
            [later_cheap, earlier_expensive]
        )

        self.assertEqual([state.state_id for state in ordered], [1, 2])

    def test_mnew_ablation_still_uses_common_infeasible_filler_order(self):
        lower_violation = _state(
            1, old_margin=-0.1, old_is_correct=False, new_margin=-2.0
        )
        higher_violation = _state(
            2, old_margin=-0.4, old_is_correct=False, new_margin=4.0
        )

        ordered, _ = frontier.feasibility_mnew_order(
            [higher_violation, lower_violation]
        )

        self.assertEqual([state.state_id for state in ordered], [1, 2])

    def test_dynamic_frontier_uses_generation_order_as_tie_break(self):
        later = _state(1, dynamic_score=0.9, generation_order=4)
        earlier = _state(2, dynamic_score=0.9, generation_order=3)
        lower = _state(3, dynamic_score=0.1, generation_order=0)

        selected, metadata = frontier.select_frontier(
            [later, lower, earlier], beam_size=2, ranking="dynamic"
        )

        self.assertEqual([state.state_id for state in selected], [2, 1])
        self.assertEqual(metadata, {})

    def test_hard_filter_discards_old_model_prediction_errors(self):
        correct = _state(1, old_is_correct=True)
        wrong = _state(2, old_is_correct=False)

        retained = frontier.strictly_feasible_states([wrong, correct])

        self.assertEqual([state.state_id for state in retained], [1])


if __name__ == "__main__":
    unittest.main()
