"""Explicit classification output contracts used by differential attacks.

The original TextAttack fork guessed whether a score tensor contained logits
or probabilities.  Differential objectives need the original logits when they
are available, so wrappers now declare the score type instead of relying on a
numeric heuristic.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np


ScoreType = Literal["logits", "probabilities"]
VALID_SCORE_TYPES = frozenset(("logits", "probabilities"))


def _as_numpy(scores: Any) -> np.ndarray:
    """Move tensor-like scores to CPU and return a detached NumPy array."""

    if hasattr(scores, "detach"):
        scores = scores.detach()
    if hasattr(scores, "cpu"):
        scores = scores.cpu()
    if hasattr(scores, "numpy"):
        scores = scores.numpy()
    return np.asarray(scores, dtype=np.float64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute a stable softmax without requiring PyTorch at analysis time."""

    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


@dataclass(frozen=True)
class ClassificationModelOutput:
    """Scores for one classification input with an explicit representation."""

    scores: np.ndarray
    score_type: ScoreType

    def __post_init__(self) -> None:
        scores = _as_numpy(self.scores)
        if self.score_type not in VALID_SCORE_TYPES:
            raise ValueError(
                "score_type must be either 'logits' or 'probabilities', "
                f"got {self.score_type!r}"
            )
        if scores.ndim != 1 or scores.size < 2:
            raise ValueError(
                "Classification scores must be a one-dimensional array with "
                f"at least two labels, got shape {scores.shape}."
            )
        if not np.isfinite(scores).all():
            raise ValueError("Classification scores must contain only finite values.")
        if self.score_type == "probabilities":
            if (scores < 0).any() or (scores > 1).any():
                raise ValueError("Probabilities must lie in the closed interval [0, 1].")
            if not np.isclose(scores.sum(), 1.0, atol=1e-6):
                raise ValueError("Probability scores must sum to 1.")

        # Frozen dataclasses still permit normalization during construction.
        object.__setattr__(self, "scores", scores)

    @property
    def probabilities(self) -> np.ndarray:
        """Return probabilities while preserving declared logits separately."""

        if self.score_type == "probabilities":
            return self.scores.copy()
        return _softmax(self.scores)

    @property
    def logits(self) -> Optional[np.ndarray]:
        """Return logits only when the wrapper explicitly supplied them."""

        if self.score_type == "logits":
            return self.scores.copy()
        return None

    @property
    def predicted_label(self) -> int:
        """Return the model prediction using NumPy's stable first-max rule."""

        return int(np.argmax(self.scores))

    @property
    def num_labels(self) -> int:
        return int(self.scores.size)

    def margin(self, label: int, *, label_should_win: bool) -> float:
        """Return a signed one-vs-best-other classification margin.

        Log-probabilities are used when only probabilities are available. This
        mirrors a logit difference while avoiding an unsupported inverse
        softmax.
        """

        if not 0 <= label < self.num_labels:
            raise ValueError(
                f"Label {label} is outside the valid range [0, {self.num_labels})."
            )
        values = (
            self.scores
            if self.score_type == "logits"
            else np.log(self.scores + 1e-12)
        )
        best_other = float(np.max(np.delete(values, label)))
        label_value = float(values[label])
        if label_should_win:
            return label_value - best_other
        return best_other - label_value

    def to_serializable(self) -> Dict[str, Any]:
        """Return a JSON-safe representation for result logs."""

        return {
            "score_type": self.score_type,
            "scores": self.scores.tolist(),
            "probabilities": self.probabilities.tolist(),
            "predicted_label": self.predicted_label,
        }


def split_classification_batch(
    scores: Any, score_type: ScoreType, expected_batch_size: int
) -> List[ClassificationModelOutput]:
    """Validate a batched wrapper result and split it into immutable rows."""

    batch = _as_numpy(scores)
    if batch.ndim == 1:
        if expected_batch_size != 1:
            raise ValueError(
                f"Received one score row for a batch of {expected_batch_size} inputs."
            )
        batch = batch.reshape(1, -1)
    if batch.ndim != 2 or batch.shape[0] != expected_batch_size:
        raise ValueError(
            "Classification wrapper returned shape "
            f"{batch.shape} for {expected_batch_size} inputs."
        )
    return [
        ClassificationModelOutput(row, score_type=score_type) for row in batch
    ]


def require_wrapper_score_type(wrapper: Any) -> ScoreType:
    """Read and validate the explicit score type declared by a wrapper."""

    score_type = getattr(wrapper, "classification_score_type", None)
    if score_type not in VALID_SCORE_TYPES:
        raise ValueError(
            f"{type(wrapper).__name__} must explicitly declare "
            "classification_score_type as 'logits' or 'probabilities' for "
            "differential classification."
        )
    return score_type


def validate_output_pair(
    new_outputs: Iterable[ClassificationModelOutput],
    old_outputs: Iterable[ClassificationModelOutput],
) -> None:
    """Ensure two model batches expose the same classification label space."""

    for new_output, old_output in zip(new_outputs, old_outputs):
        if new_output.num_labels != old_output.num_labels:
            raise ValueError(
                "The new and old models returned different numbers of labels: "
                f"{new_output.num_labels} and {old_output.num_labels}."
            )
