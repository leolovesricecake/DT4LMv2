"""Pure ranking policies for bounded differential-search frontiers.

This module deliberately depends only on the small search-state protocol. It
keeps feasibility-first ordering, Pareto ranking, and deterministic tie-breaks
testable without importing model or TextAttack runtime classes.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class FrontierRank:
    """Feasibility and ranking metadata for one current search state."""

    feasible: bool
    violation: float
    pareto_rank: int = None
    crowding_distance: float = 0.0


def _validate_states(states: Sequence) -> None:
    """Validate the minimal state fields required by every ranking policy."""

    state_ids = set()
    for state in states:
        if state.state_id in state_ids:
            raise ValueError(f"Duplicate state_id in frontier: {state.state_id!r}.")
        state_ids.add(state.state_id)
        if not isinstance(getattr(state, "old_is_correct", None), bool):
            raise TypeError("Frontier states require boolean old_is_correct values.")
        for name in (
            "old_correct_margin",
            "new_error_margin",
            "modification_cost",
        ):
            value = float(getattr(state, name))
            if not math.isfinite(value):
                raise ValueError(f"State {state.state_id!r} has non-finite {name}.")
        if float(state.modification_cost) < 0.0:
            raise ValueError("modification_cost cannot be negative.")


def infeasible_violation(state) -> float:
    """Measure distance from the old-model decision boundary for tie-breaking."""

    if state.old_is_correct:
        return 0.0
    return max(0.0, -float(state.old_correct_margin))


def strictly_feasible_states(states: Sequence) -> List:
    """Keep only states where the old model actually predicts the true label."""

    states = list(states)
    _validate_states(states)
    return [state for state in states if state.old_is_correct]


def _dominates(left, right) -> bool:
    """Return whether left dominates right on new margin and modification cost."""

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
    """Partition feasible states into deterministic Pareto fronts."""

    states = list(states)
    _validate_states(states)
    domination_counts = {state.state_id: 0 for state in states}
    dominated = {state.state_id: [] for state in states}
    state_by_id = {state.state_id: state for state in states}

    # Frontier sizes are deliberately small, making the transparent O(n^2)
    # sorter easier to audit than a specialized implementation.
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
            set(next_ids), key=lambda value: state_by_id[value].generation_order
        )
    return fronts


def crowding_distances(front: Sequence) -> Dict[int, float]:
    """Calculate normalized NSGA-II crowding distance for one Pareto front."""

    front = list(front)
    _validate_states(front)
    distances = {state.state_id: 0.0 for state in front}
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
        objective_range = maximum - minimum
        if objective_range == 0.0:
            continue
        distances[ordered[0].state_id] = float("inf")
        distances[ordered[-1].state_id] = float("inf")
        for index in range(1, len(ordered) - 1):
            state_id = ordered[index].state_id
            if math.isinf(distances[state_id]):
                continue
            previous_value = objective(ordered[index - 1])
            next_value = objective(ordered[index + 1])
            distances[state_id] += (next_value - previous_value) / objective_range
    return distances


def _infeasible_key(state):
    """Order frontier fillers by violation, new margin, cost, and stability."""

    return (
        infeasible_violation(state),
        -float(state.new_error_margin),
        float(state.modification_cost),
        state.generation_order,
    )


def feasibility_pareto_order(
    states: Sequence,
) -> Tuple[List, Dict[int, FrontierRank]]:
    """Order feasible Pareto fronts before minimum-violation fillers."""

    states = list(states)
    _validate_states(states)
    feasible = [state for state in states if state.old_is_correct]
    infeasible = [state for state in states if not state.old_is_correct]
    metadata = {
        state.state_id: FrontierRank(False, infeasible_violation(state))
        for state in infeasible
    }
    ordered_feasible = []
    for rank, front in enumerate(non_dominated_fronts(feasible), start=1):
        distances = crowding_distances(front)
        for state in front:
            metadata[state.state_id] = FrontierRank(
                True,
                0.0,
                pareto_rank=rank,
                crowding_distance=distances[state.state_id],
            )
        ordered_feasible.extend(
            sorted(
                front,
                key=lambda state: (
                    -distances[state.state_id],
                    -float(state.new_error_margin),
                    float(state.modification_cost),
                    state.generation_order,
                ),
            )
        )
    return ordered_feasible + sorted(infeasible, key=_infeasible_key), metadata


def feasibility_mnew_order(
    states: Sequence,
) -> Tuple[List, Dict[int, FrontierRank]]:
    """Rank feasible states only by new-model error margin.

    Modification cost is intentionally absent from feasible-state tie-breaking;
    this policy is the controlled ablation for FF-PBS's second Pareto objective.
    Infeasible fillers retain the common FF-PBS ordering.
    """

    states = list(states)
    _validate_states(states)
    feasible = [state for state in states if state.old_is_correct]
    infeasible = [state for state in states if not state.old_is_correct]
    ordered_feasible = sorted(
        feasible,
        key=lambda state: (-float(state.new_error_margin), state.generation_order),
    )
    best_margin = (
        float(ordered_feasible[0].new_error_margin) if ordered_feasible else None
    )
    metadata = {
        state.state_id: FrontierRank(
            True,
            0.0,
            pareto_rank=(
                1 if float(state.new_error_margin) == best_margin else 2
            ),
        )
        for state in feasible
    }
    metadata.update(
        {
            state.state_id: FrontierRank(False, infeasible_violation(state))
            for state in infeasible
        }
    )
    return ordered_feasible + sorted(infeasible, key=_infeasible_key), metadata


def dynamic_order(states: Sequence) -> List:
    """Order asynchronous states by the original DT4LM dynamic scalar score."""

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
) -> Tuple[List, Dict[int, FrontierRank]]:
    """Select at most ``beam_size`` states under one configured policy."""

    if beam_size <= 0:
        raise ValueError("beam_size must be positive.")
    if ranking == "dynamic":
        return dynamic_order(states)[:beam_size], {}
    if ranking == "feasibility_pareto":
        ordered, metadata = feasibility_pareto_order(states)
        return ordered[:beam_size], metadata
    if ranking == "feasibility_mnew":
        ordered, metadata = feasibility_mnew_order(states)
        return ordered[:beam_size], metadata
    raise ValueError(f"Unknown frontier ranking {ranking!r}.")
