"""Asynchronous bounded-frontier search for DT4LM model-pair attacks."""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods.search_method import SearchMethod

from .differential_frontier import (
    constraint_violation,
    epsilon_at,
    estimate_initial_epsilon,
    select_frontier,
    strictly_feasible_states,
    summarize_values,
)


@dataclass(frozen=True)
class DifferentialQueryEvaluation:
    """Path-independent model-pair evaluation cached by canonical input text."""

    result_type: type
    raw_output: object
    output: object
    goal_status: int
    score: float
    ground_truth_output: int
    new_model_output: Mapping
    old_model_output: Mapping
    objective_name: Optional[str]

    @classmethod
    def from_result(cls, result):
        """Freeze the path-independent fields of one queried result."""

        new_output = getattr(result, "new_model_output", None)
        old_output = getattr(result, "old_model_output", None)
        if not isinstance(new_output, dict) or not isinstance(old_output, dict):
            raise TypeError(
                "Async differential search requires serialized new/old model outputs."
            )
        return cls(
            result_type=type(result),
            raw_output=result.raw_output,
            output=result.output,
            goal_status=result.goal_status,
            score=float(result.score),
            ground_truth_output=int(result.ground_truth_output),
            new_model_output=dict(new_output),
            old_model_output=dict(old_output),
            objective_name=getattr(result, "objective_name", None),
        )

    def materialize(self, attacked_text, num_queries):
        """Create a result bound to one path-specific ``AttackedText``."""

        result = self.result_type(
            attacked_text,
            self.raw_output,
            self.output,
            self.goal_status,
            self.score,
            num_queries,
            self.ground_truth_output,
        )
        result.new_model_output = dict(self.new_model_output)
        result.old_model_output = dict(self.old_model_output)
        result.objective_name = self.objective_name
        return result


@dataclass
class DifferentialSearchState:
    """One path-aware state retained by asynchronous differential search."""

    state_id: int
    query_key: Tuple
    state_key: Tuple
    result: object
    old_prediction: int
    new_prediction: int
    old_is_correct: bool
    old_correct_margin: float
    new_error_margin: float
    modification_cost: float
    modified_indices: frozenset
    parent_id: Optional[int]
    root_child_id: Optional[int]
    depth: int
    generation_order: int
    dynamic_score: float
    expanded: bool = False
    path_has_negative_old_margin: bool = False
    path_has_old_prediction_error: bool = False
    path_has_post_root_negative_old_margin: bool = False
    path_has_post_root_old_prediction_error: bool = False
    root_dynamic_rank: Optional[int] = None


@dataclass
class SearchDiagnosticsAccumulator:
    """Accumulate compact per-sample search diagnostics online."""

    generated_candidates: int = 0
    constraint_passed_candidates: int = 0
    duplicate_states: int = 0
    query_cache_hits: int = 0
    query_cache_misses: int = 0
    queried_candidates: int = 0
    budget_truncated_candidates: int = 0
    discarded_infeasible_states: int = 0
    evaluated_states: int = 0
    expansions: int = 0
    max_depth: int = 0
    frontier_observations: int = 0
    frontier_size_sum: int = 0
    frontier_size_max: int = 0
    rank1_size_sum: int = 0
    rank1_size_max: int = 0
    modified_set_diversity_sum: int = 0
    modified_set_diversity_max: int = 0
    depth_diversity_sum: int = 0
    depth_diversity_max: int = 0
    first_expansion_old_margins: List[float] = field(default_factory=list)

    def observe_frontier(self, frontier, ranking, metadata):
        """Update online width and diversity statistics for one frontier."""

        if not frontier:
            return
        size = len(frontier)
        if ranking == "epsilon_pareto":
            rank1_size = sum(
                metadata[state.state_id].feasible
                and metadata[state.state_id].pareto_rank == 1
                for state in frontier
            )
        else:
            best_score = max(state.dynamic_score for state in frontier)
            rank1_size = sum(state.dynamic_score == best_score for state in frontier)
        modified_diversity = len({state.modified_indices for state in frontier})
        depth_diversity = len({state.depth for state in frontier})

        self.frontier_observations += 1
        self.frontier_size_sum += size
        self.frontier_size_max = max(self.frontier_size_max, size)
        self.rank1_size_sum += rank1_size
        self.rank1_size_max = max(self.rank1_size_max, rank1_size)
        self.modified_set_diversity_sum += modified_diversity
        self.modified_set_diversity_max = max(
            self.modified_set_diversity_max, modified_diversity
        )
        self.depth_diversity_sum += depth_diversity
        self.depth_diversity_max = max(self.depth_diversity_max, depth_diversity)

    @staticmethod
    def _mean(total, count):
        return total / count if count else None

    def to_serializable(
        self,
        *,
        search,
        root_state,
        epsilon_0,
        epsilon_initialization_expansion,
        epsilon_zero_initialization,
        final_epsilon,
        termination_reason,
        successful_state,
    ):
        """Build the frozen JSON-safe per-sample diagnostic object."""

        root_denominator = abs(root_state.old_correct_margin) + 1e-12
        first_summary = dict(summarize_values(self.first_expansion_old_margins))
        first_summary["negative_count"] = sum(
            value < 0.0 for value in self.first_expansion_old_margins
        )
        return {
            "method": "async_frontier",
            "ranking": search.ranking,
            "beam_size": search.beam_size,
            "epsilon_mode": search.epsilon_mode,
            "infeasible_state_policy": (
                search.infeasible_state_policy
                if search.ranking == "epsilon_pareto"
                else None
            ),
            "initial_quantile": (
                search.epsilon_initial_quantile
                if search.epsilon_mode == "adaptive"
                else None
            ),
            "initialization_max_expansions": (
                search.epsilon_initialization_max_expansions
                if search.epsilon_mode == "adaptive"
                else None
            ),
            "decay": (
                search.epsilon_decay
                if search.epsilon_mode == "adaptive"
                else None
            ),
            "root_old_margin": root_state.old_correct_margin,
            "epsilon_0": epsilon_0,
            "epsilon_to_root_margin_ratio": (
                epsilon_0 / root_denominator if epsilon_0 is not None else None
            ),
            "epsilon_initialization_expansion": epsilon_initialization_expansion,
            "epsilon_zero_initialization": epsilon_zero_initialization,
            "final_epsilon": final_epsilon,
            "expansion_count": self.expansions,
            "max_depth": self.max_depth,
            "generated_candidate_count": self.generated_candidates,
            "constraint_passed_candidate_count": self.constraint_passed_candidates,
            "candidate_state_count": self.evaluated_states,
            "duplicate_state_count": self.duplicate_states,
            "query_cache_hit_count": self.query_cache_hits,
            "query_cache_miss_count": self.query_cache_misses,
            "queried_candidate_count": self.queried_candidates,
            "budget_truncated_candidate_count": self.budget_truncated_candidates,
            "discarded_infeasible_state_count": self.discarded_infeasible_states,
            "frontier_size_mean": self._mean(
                self.frontier_size_sum, self.frontier_observations
            ),
            "frontier_size_max": self.frontier_size_max,
            "rank1_size_mean": self._mean(
                self.rank1_size_sum, self.frontier_observations
            ),
            "rank1_size_max": self.rank1_size_max,
            "frontier_modified_set_diversity_mean": self._mean(
                self.modified_set_diversity_sum, self.frontier_observations
            ),
            "frontier_modified_set_diversity_max": self.modified_set_diversity_max,
            "frontier_depth_diversity_mean": self._mean(
                self.depth_diversity_sum, self.frontier_observations
            ),
            "frontier_depth_diversity_max": self.depth_diversity_max,
            "first_expansion_old_margin": first_summary,
            "success_path_depth": (
                successful_state.depth if successful_state is not None else None
            ),
            "root_dynamic_rank": (
                successful_state.root_dynamic_rank
                if successful_state is not None
                else None
            ),
            "path_has_negative_old_margin": (
                successful_state.path_has_negative_old_margin
                if successful_state is not None
                else None
            ),
            "path_has_old_prediction_error": (
                successful_state.path_has_old_prediction_error
                if successful_state is not None
                else None
            ),
            "path_has_post_root_negative_old_margin": (
                successful_state.path_has_post_root_negative_old_margin
                if successful_state is not None
                else None
            ),
            "path_has_post_root_old_prediction_error": (
                successful_state.path_has_post_root_old_prediction_error
                if successful_state is not None
                else None
            ),
            "termination_reason": termination_reason,
        }


class AsyncDifferentialBeamSearch(SearchMethod):
    """Expand one state at a time under dynamic or epsilon-Pareto ranking."""

    def __init__(
        self,
        *,
        ranking="epsilon_pareto",
        beam_size=5,
        epsilon_mode="adaptive",
        epsilon_initial_quantile=0.75,
        epsilon_initialization_max_expansions=2,
        epsilon_decay="quadratic",
        infeasible_state_policy="feasibility_first",
        trace_output=None,
    ):
        self.ranking = ranking
        self.beam_size = int(beam_size)
        self.epsilon_mode = epsilon_mode
        self.epsilon_initial_quantile = float(epsilon_initial_quantile)
        self.epsilon_initialization_max_expansions = int(
            epsilon_initialization_max_expansions
        )
        self.epsilon_decay = epsilon_decay
        self.infeasible_state_policy = infeasible_state_policy
        self.trace_output = Path(trace_output) if trace_output else None
        self._validate_configuration()

    def _validate_configuration(self):
        if self.ranking not in {"dynamic", "epsilon_pareto"}:
            raise ValueError("ranking must be 'dynamic' or 'epsilon_pareto'.")
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive.")
        if self.epsilon_mode not in {"disabled", "strict", "adaptive"}:
            raise ValueError("Unknown epsilon mode.")
        if self.infeasible_state_policy not in {"feasibility_first", "discard"}:
            raise ValueError("Unknown infeasible-state policy.")
        if self.ranking == "dynamic" and self.epsilon_mode != "disabled":
            raise ValueError("Dynamic frontier ranking requires disabled epsilon.")
        if self.ranking == "dynamic" and self.infeasible_state_policy != (
            "feasibility_first"
        ):
            raise ValueError("Dynamic frontier ranking has no infeasible states.")
        if self.ranking == "epsilon_pareto" and self.epsilon_mode == "disabled":
            raise ValueError(
                "Epsilon-Pareto ranking requires strict or adaptive epsilon."
            )
        if self.epsilon_mode == "adaptive":
            if self.infeasible_state_policy != "feasibility_first":
                raise ValueError(
                    "Adaptive epsilon requires feasibility_first handling."
                )
            if not 0.0 <= self.epsilon_initial_quantile <= 1.0:
                raise ValueError("epsilon_initial_quantile must lie in [0, 1].")
            if self.epsilon_initialization_max_expansions <= 0:
                raise ValueError("epsilon initialization window must be positive.")
            if self.epsilon_decay not in {"linear", "quadratic"}:
                raise ValueError("epsilon_decay must be linear or quadratic.")

    @staticmethod
    def _query_key(attacked_text):
        """Identify model inputs while preserving ordered sentence-pair fields."""

        return tuple(attacked_text.text_input.items())

    @classmethod
    def _state_key(cls, attacked_text):
        """Identify future search reachability for Kuleshov word swaps."""

        attrs = attacked_text.attack_attrs
        original_index_map = tuple(int(value) for value in attrs["original_index_map"])
        return (
            cls._query_key(attacked_text),
            frozenset(int(value) for value in attrs.get("modified_indices", set())),
            original_index_map,
        )

    @classmethod
    def transformation_cache_key(cls, attacked_text, kwargs):
        """Keep cached transformations path-specific for multi-path search."""

        return ("async_frontier", cls._state_key(attacked_text)) + tuple(
            sorted(kwargs.items())
        )

    @staticmethod
    def _result_metrics(result):
        """Read the role-specific margins already produced by the goal function."""

        new_output = getattr(result, "new_model_output", None) or {}
        old_output = getattr(result, "old_model_output", None) or {}
        required = ("predicted_label", "objective_margin")
        if any(name not in new_output for name in required) or any(
            name not in old_output for name in required
        ):
            raise TypeError(
                "Differential results must expose predictions and objective margins."
            )
        return (
            int(old_output["predicted_label"]),
            int(new_output["predicted_label"]),
            float(old_output["objective_margin"]),
            float(new_output["objective_margin"]),
        )

    def _build_state(
        self,
        *,
        state_id,
        result,
        root_text,
        parent,
        generation_order,
    ):
        old_prediction, new_prediction, old_margin, new_margin = self._result_metrics(
            result
        )
        label = int(result.ground_truth_output)
        depth = 0 if parent is None else parent.depth + 1
        state = DifferentialSearchState(
            state_id=state_id,
            query_key=self._query_key(result.attacked_text),
            state_key=self._state_key(result.attacked_text),
            result=result,
            old_prediction=old_prediction,
            new_prediction=new_prediction,
            old_is_correct=(old_prediction == label),
            old_correct_margin=old_margin,
            new_error_margin=new_margin,
            modification_cost=result.attacked_text.modification_rate(root_text),
            modified_indices=frozenset(
                int(value)
                for value in result.attacked_text.attack_attrs.get(
                    "modified_indices", set()
                )
            ),
            parent_id=parent.state_id if parent is not None else None,
            root_child_id=None,
            depth=depth,
            generation_order=generation_order,
            dynamic_score=float(result.score),
            path_has_negative_old_margin=(
                old_margin < 0.0
                or (
                    parent is not None and parent.path_has_negative_old_margin
                )
            ),
            path_has_old_prediction_error=(
                old_prediction != label
                or (
                    parent is not None and parent.path_has_old_prediction_error
                )
            ),
            path_has_post_root_negative_old_margin=(
                parent is not None
                and (
                    old_margin < 0.0
                    or parent.path_has_post_root_negative_old_margin
                )
            ),
            path_has_post_root_old_prediction_error=(
                parent is not None
                and (
                    old_prediction != label
                    or parent.path_has_post_root_old_prediction_error
                )
            ),
        )
        if depth == 1:
            state.root_child_id = state.state_id
        elif parent is not None:
            state.root_child_id = parent.root_child_id
            state.root_dynamic_rank = parent.root_dynamic_rank
        return state

    def _query_budget(self):
        budget = self.goal_function.query_budget
        if not math.isfinite(budget) or int(budget) <= 0:
            raise ValueError(
                "Async differential search requires a finite positive query budget."
            )
        return int(budget)

    def _current_epsilon(self, epsilon_0):
        if self.epsilon_mode in {"disabled", "strict"} or epsilon_0 is None:
            return 0.0
        return epsilon_at(
            epsilon_0,
            self.goal_function.num_queries,
            self._query_budget(),
            self.epsilon_decay,
        )

    def _rank_root_children(self, states):
        """Attach stable one-based dynamic ranks to first-step states."""

        ordered = sorted(
            states,
            key=lambda state: (-state.dynamic_score, state.generation_order),
        )
        for rank, state in enumerate(ordered, start=1):
            state.root_dynamic_rank = rank

    @staticmethod
    def _select_success(states):
        successful = [
            state
            for state in states
            if state.result.goal_status == GoalFunctionResultStatus.SUCCEEDED
        ]
        if not successful:
            return None
        return min(
            successful,
            key=lambda state: (
                state.modification_cost,
                -state.new_error_margin,
                -state.old_correct_margin,
                state.generation_order,
            ),
        )

    def _select_states(self, states, epsilon):
        return select_frontier(
            states,
            self.beam_size,
            self.ranking,
            epsilon,
        )

    def _failure_state(self, states, epsilon):
        selectable = [
            state
            for state in states
            if state.result.goal_status != GoalFunctionResultStatus.SUCCEEDED
        ]
        if not selectable:
            return states[0]
        if self.infeasible_state_policy == "discard":
            strictly_feasible = strictly_feasible_states(selectable)
            if strictly_feasible:
                selectable = strictly_feasible
            else:
                return states[0]
        selected, _ = self._select_states(selectable, epsilon)
        return selected[0]

    def _write_trace(
        self,
        root_state,
        parent,
        frontier,
        epsilon,
        new_states,
        metadata,
        parent_metadata,
        parent_selection_epsilon,
        batch_stats,
    ):
        if self.trace_output is None:
            return
        self.trace_output.parent.mkdir(parents=True, exist_ok=True)
        if self.ranking == "epsilon_pareto":
            rank1_size = sum(
                metadata[state.state_id].feasible
                and metadata[state.state_id].pareto_rank == 1
                for state in frontier
            )
            parent_rank = (
                parent_metadata[parent.state_id].pareto_rank
                if parent.state_id in parent_metadata
                and parent_metadata[parent.state_id].feasible
                else None
            )
        else:
            best_score = max(
                (state.dynamic_score for state in frontier), default=None
            )
            rank1_size = sum(
                state.dynamic_score == best_score for state in frontier
            )
            parent_rank = 1
        record = {
            "dataset_index": root_state.result.attacked_text.attack_attrs.get(
                "dataset_index"
            ),
            "expansion_index": self._diagnostics.expansions,
            "parent_state_id": parent.state_id,
            "parent_root_child_id": parent.root_child_id,
            "parent_depth": parent.depth,
            "query_count": self.goal_function.num_queries,
            "query_budget": self._query_budget(),
            "epsilon": epsilon,
            "parent_selection_epsilon": parent_selection_epsilon,
            "frontier_size": len(frontier),
            "rank1_size": rank1_size,
            "evaluated_state_count": len(new_states),
            "successful_state_count": sum(
                state.result.goal_status == GoalFunctionResultStatus.SUCCEEDED
                for state in new_states
            ),
            "parent_old_margin": parent.old_correct_margin,
            "parent_new_margin": parent.new_error_margin,
            "parent_modification_cost": parent.modification_cost,
            "parent_constraint_violation": constraint_violation(
                parent.old_correct_margin, parent_selection_epsilon
            ),
            "parent_rank": parent_rank,
            **batch_stats,
        }
        with open(self.trace_output, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _attach_diagnostics(
        self,
        result,
        *,
        root_state,
        epsilon_0,
        initialization_expansion,
        epsilon_zero_initialization,
        termination_reason,
        successful_state=None,
    ):
        final_epsilon = self._current_epsilon(epsilon_0)
        result.search_diagnostics = self._diagnostics.to_serializable(
            search=self,
            root_state=root_state,
            epsilon_0=epsilon_0,
            epsilon_initialization_expansion=initialization_expansion,
            epsilon_zero_initialization=epsilon_zero_initialization,
            final_epsilon=final_epsilon,
            termination_reason=termination_reason,
            successful_state=successful_state,
        )
        return result

    def perform_search(self, initial_result):
        """Run one deterministic asynchronous search for an attack sample."""

        objective_name = getattr(initial_result, "objective_name", None)
        if objective_name != "dynamic":
            raise ValueError(
                "Async differential search currently requires the dynamic "
                "compatibility objective."
            )

        root_text = initial_result.attacked_text
        root_state = self._build_state(
            state_id=0,
            result=initial_result,
            root_text=root_text,
            parent=None,
            generation_order=0,
        )
        query_cache = {
            root_state.query_key: DifferentialQueryEvaluation.from_result(
                initial_result
            )
        }
        seen_state_keys = {root_state.state_key}
        all_states = [root_state]
        frontier = [root_state]
        next_state_id = 1
        next_generation_order = 1
        epsilon_0 = 0.0 if self.epsilon_mode == "strict" else None
        initialization_expansion = 0 if self.epsilon_mode == "strict" else None
        epsilon_zero_initialization = (
            True if self.epsilon_mode == "strict" else None
        )
        initialization_query_keys = set()
        initialization_violations = []
        self._diagnostics = SearchDiagnosticsAccumulator()
        termination_reason = "frontier_empty"

        query_budget = self._query_budget()
        while frontier and self.goal_function.num_queries < query_budget:
            epsilon = self._current_epsilon(epsilon_0)
            parent_selection_epsilon = epsilon
            ordered, parent_metadata = self._select_states(frontier, epsilon)
            parent = ordered[0]
            frontier.remove(parent)
            parent.expanded = True
            self._diagnostics.expansions += 1

            transformations = self.get_transformations(
                parent.result.attacked_text, original_text=root_text
            )
            transformation_stats = (
                self.get_last_transformation_stats()
                if hasattr(self, "get_last_transformation_stats")
                else {}
            )
            self._diagnostics.generated_candidates += int(
                transformation_stats.get("generated", len(transformations))
            )
            self._diagnostics.constraint_passed_candidates += len(transformations)
            for candidate_order, candidate in enumerate(transformations):
                candidate.attack_attrs["search_round"] = (
                    self._diagnostics.expansions - 1
                )
                candidate.attack_attrs["candidate_order"] = candidate_order

            candidate_entries = []
            local_state_keys = set()
            duplicate_count_before = self._diagnostics.duplicate_states
            for candidate in transformations:
                state_key = self._state_key(candidate)
                if state_key in seen_state_keys or state_key in local_state_keys:
                    self._diagnostics.duplicate_states += 1
                    continue
                local_state_keys.add(state_key)
                candidate_entries.append(
                    (candidate, self._query_key(candidate), state_key)
                )

            miss_entries = []
            miss_query_keys = set()
            for candidate, query_key, _ in candidate_entries:
                if query_key in query_cache or query_key in miss_query_keys:
                    continue
                miss_query_keys.add(query_key)
                miss_entries.append((candidate, query_key))

            self._diagnostics.query_cache_misses += len(miss_entries)
            queried_results, search_over = self.get_goal_results(
                [candidate for candidate, _ in miss_entries]
            )
            self._diagnostics.queried_candidates += len(queried_results)
            self._diagnostics.budget_truncated_candidates += (
                len(miss_entries) - len(queried_results)
            )
            for result, (_, query_key) in zip(queried_results, miss_entries):
                query_cache[query_key] = DifferentialQueryEvaluation.from_result(
                    result
                )
            queried_representative_ids = {
                id(candidate)
                for candidate, _ in miss_entries[: len(queried_results)]
            }

            new_states = []
            for candidate, query_key, state_key in candidate_entries:
                evaluation = query_cache.get(query_key)
                if evaluation is None:
                    continue
                if id(candidate) not in queried_representative_ids:
                    self._diagnostics.query_cache_hits += 1
                result = evaluation.materialize(
                    candidate, self.goal_function.num_queries
                )
                state = self._build_state(
                    state_id=next_state_id,
                    result=result,
                    root_text=root_text,
                    parent=parent,
                    generation_order=next_generation_order,
                )
                next_state_id += 1
                next_generation_order += 1
                seen_state_keys.add(state_key)
                new_states.append(state)
                all_states.append(state)
                self._diagnostics.max_depth = max(
                    self._diagnostics.max_depth, state.depth
                )
            self._diagnostics.evaluated_states += len(new_states)
            retained_new_states = new_states
            discarded_infeasible_states = 0
            if self.infeasible_state_policy == "discard":
                retained_new_states = strictly_feasible_states(new_states)
                discarded_infeasible_states = (
                    len(new_states) - len(retained_new_states)
                )
                self._diagnostics.discarded_infeasible_states += (
                    discarded_infeasible_states
                )
            batch_stats = {
                "generated_candidate_count": int(
                    transformation_stats.get("generated", len(transformations))
                ),
                "constraint_passed_candidate_count": len(transformations),
                "duplicate_state_count": (
                    self._diagnostics.duplicate_states - duplicate_count_before
                ),
                "query_cache_miss_count": len(miss_entries),
                "queried_candidate_count": len(queried_results),
                "query_cache_hit_count": max(
                    0, len(new_states) - len(queried_results)
                ),
                "budget_truncated_candidate_count": (
                    len(miss_entries) - len(queried_results)
                ),
                "discarded_infeasible_state_count": discarded_infeasible_states,
            }

            if parent.depth == 0:
                self._rank_root_children(new_states)
                unique_first_margins = {}
                for state in new_states:
                    unique_first_margins.setdefault(
                        state.query_key, state.old_correct_margin
                    )
                self._diagnostics.first_expansion_old_margins = list(
                    unique_first_margins.values()
                )

            if self.epsilon_mode == "adaptive" and epsilon_0 is None:
                for state in new_states:
                    if state.query_key in initialization_query_keys:
                        continue
                    initialization_query_keys.add(state.query_key)
                    if state.old_correct_margin < 0.0:
                        initialization_violations.append(-state.old_correct_margin)
                if initialization_violations:
                    epsilon_0 = estimate_initial_epsilon(
                        initialization_violations,
                        self.epsilon_initial_quantile,
                    )
                    initialization_expansion = self._diagnostics.expansions
                    epsilon_zero_initialization = False
                elif (
                    self._diagnostics.expansions
                    >= self.epsilon_initialization_max_expansions
                ):
                    epsilon_0 = 0.0
                    initialization_expansion = self._diagnostics.expansions
                    epsilon_zero_initialization = True

            successful_state = self._select_success(new_states)
            if successful_state is not None:
                if self.epsilon_mode == "adaptive" and epsilon_0 is None:
                    # No later expansion can initialize epsilon after an early
                    # success, so freeze the observed no-violation case as zero.
                    epsilon_0 = 0.0
                    initialization_expansion = self._diagnostics.expansions
                    epsilon_zero_initialization = True
                trace_frontier, trace_metadata = self._select_states(
                    frontier + retained_new_states,
                    self._current_epsilon(epsilon_0),
                )
                self._write_trace(
                    root_state,
                    parent,
                    trace_frontier,
                    self._current_epsilon(epsilon_0),
                    new_states,
                    trace_metadata,
                    parent_metadata,
                    parent_selection_epsilon,
                    batch_stats,
                )
                return self._attach_diagnostics(
                    successful_state.result,
                    root_state=root_state,
                    epsilon_0=epsilon_0,
                    initialization_expansion=initialization_expansion,
                    epsilon_zero_initialization=epsilon_zero_initialization,
                    termination_reason="success",
                    successful_state=successful_state,
                )

            epsilon = self._current_epsilon(epsilon_0)
            frontier, metadata = self._select_states(
                frontier + retained_new_states, epsilon
            )
            self._diagnostics.observe_frontier(frontier, self.ranking, metadata)
            self._write_trace(
                root_state,
                parent,
                frontier,
                epsilon,
                new_states,
                metadata,
                parent_metadata,
                parent_selection_epsilon,
                batch_stats,
            )
            if search_over:
                termination_reason = "budget_exhausted"
                break
            if not transformations and not frontier:
                termination_reason = "no_transformations"
                break

        if self.epsilon_mode == "adaptive" and epsilon_0 is None:
            epsilon_0 = 0.0
            initialization_expansion = self._diagnostics.expansions
            epsilon_zero_initialization = True
        if self.goal_function.num_queries >= query_budget:
            termination_reason = "budget_exhausted"
        epsilon = self._current_epsilon(epsilon_0)
        failure_state = self._failure_state(all_states, epsilon)
        return self._attach_diagnostics(
            failure_state.result,
            root_state=root_state,
            epsilon_0=epsilon_0,
            initialization_expansion=initialization_expansion,
            epsilon_zero_initialization=epsilon_zero_initialization,
            termination_reason=termination_reason,
        )

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return [
            "ranking",
            "beam_size",
            "epsilon_mode",
            "epsilon_initial_quantile",
            "epsilon_initialization_max_expansions",
            "epsilon_decay",
            "infeasible_state_policy",
        ]
