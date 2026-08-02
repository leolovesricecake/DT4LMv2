"""Pure AE-PBS tests that avoid importing the complete TextAttack package."""

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
    """Build the minimal state protocol consumed by pure ranking functions."""

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
    def test_linear_quantile_and_compact_summary_are_deterministic(self):
        self.assertAlmostEqual(frontier.linear_quantile([0.0, 10.0], 0.25), 2.5)
        self.assertEqual(frontier.linear_quantile([3.0], 0.75), 3.0)
        self.assertEqual(
            frontier.summarize_values([4.0, 1.0, 3.0, 2.0]),
            {
                "count": 4,
                "min": 1.0,
                "q1": 1.75,
                "median": 2.5,
                "q3": 3.25,
                "max": 4.0,
            },
        )
        with self.assertRaises(ValueError):
            frontier.linear_quantile([], 0.5)

    def test_epsilon_estimation_and_decay_cover_budget_boundary(self):
        self.assertEqual(frontier.estimate_initial_epsilon([], 0.75), 0.0)
        self.assertAlmostEqual(
            frontier.estimate_initial_epsilon([1.0, 3.0], 0.75), 2.5
        )
        self.assertEqual(frontier.epsilon_at(4.0, 0, 100, "quadratic"), 4.0)
        self.assertEqual(frontier.epsilon_at(4.0, 50, 100, "quadratic"), 1.0)
        self.assertEqual(frontier.epsilon_at(4.0, 50, 100, "linear"), 2.0)
        self.assertEqual(frontier.epsilon_at(4.0, 100, 100, "quadratic"), 0.0)
        self.assertEqual(frontier.epsilon_at(4.0, 120, 100, "quadratic"), 0.0)

    def test_non_dominated_fronts_preserve_margin_cost_tradeoffs(self):
        cheap = _state(1, new_margin=0.1, cost=0.1)
        aggressive = _state(2, new_margin=0.8, cost=0.4)
        dominated = _state(3, new_margin=0.0, cost=0.5)

        fronts = frontier.non_dominated_fronts([cheap, aggressive, dominated])

        self.assertEqual([state.state_id for state in fronts[0]], [1, 2])
        self.assertEqual([state.state_id for state in fronts[1]], [3])

    def test_feasible_candidates_precede_infeasible_violations(self):
        feasible = _state(1, old_margin=-0.1, new_margin=0.1, cost=0.2)
        close = _state(2, old_margin=-0.21, new_margin=1.0, cost=0.1)
        far = _state(3, old_margin=-0.8, new_margin=2.0, cost=0.1)

        ordered, metadata = frontier.epsilon_pareto_order(
            [far, close, feasible], epsilon=0.2
        )

        self.assertEqual([state.state_id for state in ordered], [1, 2, 3])
        self.assertTrue(metadata[1].feasible)
        self.assertAlmostEqual(metadata[2].violation, 0.01)
        self.assertAlmostEqual(metadata[3].violation, 0.6)

    def test_crowding_distance_and_generation_order_are_stable(self):
        states = [
            _state(1, new_margin=0.0, cost=0.1, generation_order=2),
            _state(2, new_margin=0.5, cost=0.2, generation_order=1),
            _state(3, new_margin=1.0, cost=0.3, generation_order=0),
        ]
        selected, metadata = frontier.select_frontier(
            states, beam_size=2, ranking="epsilon_pareto", epsilon=0.0
        )

        self.assertEqual([state.state_id for state in selected], [3, 1])
        self.assertEqual(metadata[1].crowding_distance, float("inf"))
        self.assertEqual(metadata[3].crowding_distance, float("inf"))

    def test_dynamic_frontier_uses_generation_order_as_tie_break(self):
        later = _state(1, dynamic_score=0.9, generation_order=4)
        earlier = _state(2, dynamic_score=0.9, generation_order=3)
        lower = _state(3, dynamic_score=0.1, generation_order=0)

        selected, metadata = frontier.select_frontier(
            [later, lower, earlier], beam_size=2, ranking="dynamic"
        )

        self.assertEqual([state.state_id for state in selected], [2, 1])
        self.assertEqual(metadata, {})

    def test_strict_feasibility_discards_old_model_errors(self):
        correct = _state(1, old_is_correct=True)
        wrong = _state(2, old_is_correct=False)

        retained = frontier.strictly_feasible_states([wrong, correct])

        self.assertEqual([state.state_id for state in retained], [1])


if __name__ == "__main__":
    unittest.main()
