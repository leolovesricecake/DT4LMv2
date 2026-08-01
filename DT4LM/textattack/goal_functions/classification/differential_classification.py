"""Differential classification goal with pluggable DT4LM objectives."""

from .classification_goal_function import ClassificationGoalFunction
from .differential_objectives import (
    create_differential_objective,
    differential_success,
)
from textattack.models.classification_output import (
    ClassificationModelOutput,
    require_wrapper_score_type,
    split_classification_batch,
)


class DifferentialClassification(ClassificationGoalFunction):
    """Generate candidates where the old model stays correct and new regresses."""

    def __init__(self, *args, objective="dynamic", **kwargs):
        super().__init__(*args, **kwargs)
        if self.model2 is None:
            raise ValueError("DifferentialClassification requires two model wrappers.")
        self.objective = (
            create_differential_objective(objective)
            if isinstance(objective, str)
            else objective
        )
        # Fail before the attack starts if either wrapper leaves score semantics
        # ambiguous. No numeric inference is allowed in pair mode.
        require_wrapper_score_type(self.model)
        require_wrapper_score_type(self.model2)

    def _process_model_outputs_for_wrapper(self, inputs, scores, model_wrapper):
        score_type = require_wrapper_score_type(model_wrapper)
        return split_classification_batch(scores, score_type, len(inputs))

    def _process_model_outputs(self, inputs, scores):
        """Pair mode always routes through the wrapper-aware processing hook."""

        raise RuntimeError(
            "Differential outputs must be processed with their originating wrapper."
        )

    def _is_goal_complete(
        self,
        new_output: ClassificationModelOutput,
        old_output: ClassificationModelOutput,
        _,
    ):
        self._validate_label(new_output, old_output)
        return differential_success(
            new_output, old_output, int(self.ground_truth_output)
        )

    def _should_skip_2(
        self,
        new_output: ClassificationModelOutput,
        old_output: ClassificationModelOutput,
        _,
    ):
        """Skip only inputs that already satisfy the differential goal."""

        self._validate_label(new_output, old_output)
        return differential_success(
            new_output, old_output, int(self.ground_truth_output)
        )

    def _get_score(
        self,
        new_output: ClassificationModelOutput,
        old_output: ClassificationModelOutput,
        _args,
        attacked_text,
    ):
        self._validate_label(new_output, old_output)
        modification_cost = attacked_text.modification_rate(
            self.initial_attacked_text
        )
        return self.objective.score(
            new_output,
            old_output,
            int(self.ground_truth_output),
            modification_cost,
        )

    def _get_displayed_output(self, raw_output):
        return raw_output.predicted_label

    def _validate_label(self, new_output, old_output):
        if new_output.num_labels != old_output.num_labels:
            raise ValueError(
                "The new and old models returned different label counts: "
                f"{new_output.num_labels} and {old_output.num_labels}."
            )
        if not 0 <= int(self.ground_truth_output) < new_output.num_labels:
            raise ValueError(
                f"Ground-truth label {self.ground_truth_output} is outside the "
                f"model label range [0, {new_output.num_labels})."
            )
