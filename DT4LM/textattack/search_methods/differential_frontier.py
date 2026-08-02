"""Pure ranking utilities for differential asynchronous frontier search.

The functions in this module intentionally do not import TextAttack model or
attack classes. Keeping the numerical policy independent makes Pareto ranking,
epsilon scheduling, and deterministic tie-breaking testable in isolation.
"""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class FrontierRank:
    """Current epsilon-feasibility and Pareto metadata for one search state."""

    feasible: bool
    violation: float
    pareto_rank: int = None
    crowding_distance: float = 0.0


def linear_quantile(values: Iterable[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated sample quantile."""

    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("Cannot calculate a quantile from an empty sequence.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in the closed interval [0, 1].")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Quantile values must be finite.")

    position = (len(values) - 1) * float(quantile)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return values[lower_index]
    fraction = position - lower_index
    return values[lower_index] + fraction * (
        values[upper_index] - values[lower_index]
    )


def summarize_values(values: Iterable[float]) -> Mapping[str, float]:
    """Return a compact distribution summary without retaining raw values."""

    values = [float(value) for value in values]
    if not values:
        return {
            "count": 0,
            "min": None,
            "q1": None,
            "median": None,
            "q3": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "q1": linear_quantile(values, 0.25),
        "median": linear_quantile(values, 0.50),
        "q3": linear_quantile(values, 0.75),
        "max": max(values),
    }


def estimate_initial_epsilon(violations: Iterable[float], quantile: float) -> float:
    """Estimate epsilon from strictly positive old-margin violations."""

    positive = [float(value) for value in violations if float(value) > 0.0]
    return linear_quantile(positive, quantile) if positive else 0.0


def epsilon_at(
    epsilon_0: float,
    query_count: int,
    query_budget: int,
    decay: str,
) -> float:
    """Calculate query-budget-driven linear or quadratic epsilon decay."""

    epsilon_0 = float(epsilon_0)
    if epsilon_0 < 0.0 or not math.isfinite(epsilon_0):
        raise ValueError("epsilon_0 must be finite and non-negative.")
    if query_budget <= 0:
        raise ValueError("query_budget must be positive.")
    if decay not in {"linear", "quadratic"}:
        raise ValueError("decay must be either 'linear' or 'quadratic'.")

    progress = min(max(float(query_count) / float(query_budget), 0.0), 1.0)
    remaining = 1.0 - progress
    exponent = 1 if decay == "linear" else 2
    return epsilon_0 * (remaining**exponent)


def constraint_violation(old_correct_margin: float, epsilon: float) -> float:
    """Return violation of ``old_margin >= -epsilon``."""

    old_correct_margin = float(old_correct_margin)
    epsilon = float(epsilon)
    if not math.isfinite(old_correct_margin):
        raise ValueError("old_correct_margin must be finite.")
    if epsilon < 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and non-negative.")
    return max(0.0, -epsilon - old_correct_margin)


def _validate_states(states: Sequence) -> None:
    """Validate the minimal state protocol required by frontier ranking."""

    state_ids = set()
    for state in states:
        state_id = state.state_id
        if state_id in state_ids:
            raise ValueError(f"Duplicate state_id in frontier: {state_id!r}.")
        state_ids.add(state_id)
        for name in ("old_correct_margin", "new_error_margin", "modification_cost"):
            value = float(getattr(state, name))
            if not math.isfinite(value):
                raise ValueError(f"State {state_id!r} has non-finite {name}.")
        if float(state.modification_cost) < 0.0:
            raise ValueError("modification_cost cannot be negative.")


def _dominates(left, right) -> bool:
    """Return whether left Pareto-dominates right on margin and cost."""

    no_worse = (
        left.new_error_margin >= right.new_error_margin
        and left.modification_cost <= right.modification_cost
    )
    strictly_better = (
        left.new_error_margin > right.new_error_margin
        or left.modification_cost < right.modification_cost
    )
    return bool(no_worse and strictly_better)


def non_dominated_fronts(states: Sequence) -> List[List]:
    """Partition states into deterministic Pareto fronts."""

    states = list(states)
    _validate_states(states)
    domination_counts = {state.state_id: 0 for state in states}
    dominated = {state.state_id: [] for state in states}
    state_by_id = {state.state_id: state for state in states}

    # The frontier is bounded by the query budget, so the transparent O(n^2)
    # implementation is preferable to a more fragile specialized sorter.
    for left in states:
        for right in states:
            if left.state_id == right.state_id:
                continue
            if _dominates(left, right):
                dominated[left.state_id].append(right.state_id)
            elif _dominates(right, left):
                domination_counts[left.state_id] += 1

    current_ids = [
        state.state_id for state in states if domination_counts[state.state_id] == 0
    ]
    fronts = []
    while current_ids:
        current = [state_by_id[state_id] for state_id in current_ids]
        current.sort(key=lambda state: state.generation_order)
        fronts.append(current)
        next_ids = []
        for state_id in current_ids:
            for dominated_id in dominated[state_id]:
                domination_counts[dominated_id] -= 1
                if domination_counts[dominated_id] == 0:
                    next_ids.append(dominated_id)
        current_ids = sorted(
            set(next_ids), key=lambda state_id: state_by_id[state_id].generation_order
        )
    return fronts


def crowding_distances(front: Sequence) -> Dict[int, float]:
    """Calculate normalized NSGA-II crowding distances for one front."""

    front = list(front)
    _validate_states(front)
    distances = {state.state_id: 0.0 for state in front}
    if not front:
        return distances
    if len(front) <= 2:
        return {state.state_id: float("inf") for state in front}

    objectives = (
        lambda state: float(state.new_error_margin),
        lambda state: float(state.modification_cost),
    )
    for objective in objectives:
        ordered = sorted(
            front, key=lambda state: (objective(state), state.generation_order)
        )
        minimum = objective(ordered[0])
        maximum = objective(ordered[-1])
        distances[ordered[0].state_id] = float("inf")
        distances[ordered[-1].state_id] = float("inf")
        objective_range = maximum - minimum
        if objective_range == 0.0:
            continue
        for index in range(1, len(ordered) - 1):
            state_id = ordered[index].state_id
            if math.isinf(distances[state_id]):
                continue
            previous_value = objective(ordered[index - 1])
            next_value = objective(ordered[index + 1])
            distances[state_id] += (next_value - previous_value) / objective_range
    return distances


def epsilon_pareto_order(
    states: Sequence, epsilon: float
) -> Tuple[List, Dict[int, FrontierRank]]:
    """Order states by constrained Pareto rank and deterministic tie-breaks."""

    states = list(states)
    _validate_states(states)
    feasible = []
    metadata = {}
    for state in states:
        violation = constraint_violation(state.old_correct_margin, epsilon)
        if violation == 0.0:
            feasible.append(state)
        else:
            metadata[state.state_id] = FrontierRank(False, violation)

    for rank, front in enumerate(non_dominated_fronts(feasible), start=1):
        distances = crowding_distances(front)
        for state in front:
            metadata[state.state_id] = FrontierRank(
                True,
                0.0,
                pareto_rank=rank,
                crowding_distance=distances[state.state_id],
            )

    def order_key(state):
        state_rank = metadata[state.state_id]
        if state_rank.feasible:
            return (
                0,
                state_rank.pareto_rank,
                -state_rank.crowding_distance,
                -float(state.new_error_margin),
                float(state.modification_cost),
                state.generation_order,
            )
        return (
            1,
            state_rank.violation,
            -float(state.new_error_margin),
            float(state.modification_cost),
            state.generation_order,
        )

    return sorted(states, key=order_key), metadata


def dynamic_order(states: Sequence) -> List:
    """Order asynchronous states by the legacy dynamic scalar score."""

    states = list(states)
    _validate_states(states)
    for state in states:
        if not math.isfinite(float(state.dynamic_score)):
            raise ValueError("Dynamic frontier scores must be finite.")
    return sorted(
        states,
        key=lambda state: (-float(state.dynamic_score), state.generation_order),
    )


def select_frontier(
    states: Sequence,
    beam_size: int,
    ranking: str,
    epsilon: float = 0.0,
) -> Tuple[List, Dict[int, FrontierRank]]:
    """Select at most ``beam_size`` states under one configured policy."""

    if beam_size <= 0:
        raise ValueError("beam_size must be positive.")
    if ranking == "dynamic":
        ordered = dynamic_order(states)
        return ordered[:beam_size], {}
    if ranking == "epsilon_pareto":
        ordered, metadata = epsilon_pareto_order(states, epsilon)
        return ordered[:beam_size], metadata
    raise ValueError(f"Unknown frontier ranking {ranking!r}.")
