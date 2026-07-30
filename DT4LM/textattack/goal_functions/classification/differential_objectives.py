"""Objective strategies for DT4LM differential classification."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

from textattack.models.classification_output import ClassificationModelOutput


def differential_success(
    new_output: ClassificationModelOutput,
    old_output: ClassificationModelOutput,
    label: int,
) -> bool:
    """Return true only for an old-correct/new-incorrect regression."""

    return (
        old_output.predicted_label == label
        and new_output.predicted_label != label
    )


@dataclass(frozen=True, order=True)
class LexiScore:
    """Five-part score whose native tuple ordering implements LexiDT."""

    old_is_correct: int
    old_margin: float
    new_is_incorrect: int
    new_margin: float
    negative_cost: float

    def as_tuple(self) -> Tuple[int, float, int, float, float]:
        return (
            self.old_is_correct,
            self.old_margin,
            self.new_is_incorrect,
            self.new_margin,
            self.negative_cost,
        )

    def to_serializable(self) -> Dict[str, float]:
        """Expose named components so logs do not flatten the objective."""

        return {
            "old_is_correct": self.old_is_correct,
            "old_margin": self.old_margin,
            "new_is_incorrect": self.new_is_incorrect,
            "new_margin": self.new_margin,
            "negative_cost": self.negative_cost,
        }


class DifferentialObjective(ABC):
    """Interface shared by dynamic, static, and lexicographic objectives."""

    name: str

    @abstractmethod
    def score(
        self,
        new_output: ClassificationModelOutput,
        old_output: ClassificationModelOutput,
        label: int,
        modification_cost: float,
    ):
        raise NotImplementedError()


class DynamicObjective(DifferentialObjective):
    """The original DT4LM piecewise objective, preserved as the baseline."""

    name = "dynamic"

    def score(
        self,
        new_output: ClassificationModelOutput,
        old_output: ClassificationModelOutput,
        label: int,
        modification_cost: float,
    ) -> float:
        del modification_cost
        new_prob = float(new_output.probabilities[label])
        old_prob = float(old_output.probabilities[label])
        new_correct = new_output.predicted_label == label
        old_correct = old_output.predicted_label == label
        lambda1 = 1.0 + (new_prob - 0.5)
        lambda2 = 1.0 + (0.5 - old_prob)

        # Keep the four original branches verbatim in mathematical form so
        # baseline scores remain comparable with existing DT4LM artifacts.
        if new_correct and old_correct:
            return old_prob - lambda1 * new_prob + 0.5
        if new_correct and not old_correct:
            return old_prob - new_prob
        if not new_correct and not old_correct:
            return lambda2 * old_prob - new_prob + 0.5
        return 1e6


class StaticObjective(DifferentialObjective):
    """Parameter-free old-minus-new true-label probability objective."""

    name = "static"

    def score(
        self,
        new_output: ClassificationModelOutput,
        old_output: ClassificationModelOutput,
        label: int,
        modification_cost: float,
    ) -> float:
        del modification_cost
        return float(
            old_output.probabilities[label] - new_output.probabilities[label]
        )


class LexicographicObjective(DifferentialObjective):
    """LexiDT objective with explicit prediction-state tie handling."""

    name = "lexi"

    def score(
        self,
        new_output: ClassificationModelOutput,
        old_output: ClassificationModelOutput,
        label: int,
        modification_cost: float,
    ) -> LexiScore:
        old_margin = old_output.margin(label, label_should_win=True)
        new_margin = new_output.margin(label, label_should_win=False)
        return LexiScore(
            old_is_correct=int(old_output.predicted_label == label),
            old_margin=min(old_margin, 0.0),
            new_is_incorrect=int(new_output.predicted_label != label),
            new_margin=min(new_margin, 0.0),
            negative_cost=-float(modification_cost),
        )


OBJECTIVES = {
    "dynamic": DynamicObjective,
    "static": StaticObjective,
    "lexi": LexicographicObjective,
}


def create_differential_objective(name: str) -> DifferentialObjective:
    """Create a validated objective strategy from its CLI name."""

    try:
        return OBJECTIVES[name]()
    except KeyError as exc:
        choices = ", ".join(sorted(OBJECTIVES))
        raise ValueError(
            f"Unknown differential objective {name!r}; choose one of {choices}."
        ) from exc
