"""Immutable sample manifests for comparable differential experiments."""

from dataclasses import asdict, dataclass
import json
import random
from typing import Iterable, List, Optional, Sequence

from .dataset import Dataset


@dataclass(frozen=True)
class SampleManifest:
    """The exact eligible population and selected attack sample order."""

    dataset_id: str
    dataset_revision_or_fingerprint: str
    split: str
    old_model_id: str
    new_model_id: str
    seed: int
    test_split_size: int
    eligible_indices: List[int]
    selected_indices: List[int]
    old_model_revision: Optional[str] = None
    new_model_revision: Optional[str] = None

    @property
    def eligible_count(self) -> int:
        return len(self.eligible_indices)

    @property
    def sample_count(self) -> int:
        return len(self.selected_indices)

    def validate(self) -> None:
        """Reject malformed manifests before they can alter metric denominators."""

        if self.test_split_size <= 0:
            raise ValueError("test_split_size must be positive.")
        if not self.eligible_indices:
            raise ValueError("A sample manifest must contain eligible samples.")
        if len(set(self.eligible_indices)) != len(self.eligible_indices):
            raise ValueError("eligible_indices contains duplicates.")
        if len(set(self.selected_indices)) != len(self.selected_indices):
            raise ValueError("selected_indices contains duplicates.")
        eligible = set(self.eligible_indices)
        if any(index not in eligible for index in self.selected_indices):
            raise ValueError("Every selected index must also be eligible.")
        if any(
            index < 0 or index >= self.test_split_size
            for index in self.eligible_indices
        ):
            raise ValueError("Manifest indices must lie inside the test split.")

    def to_dict(self):
        data = asdict(self)
        # Store derived counts to make experiment artifacts self-describing.
        data["eligible_count"] = self.eligible_count
        data["sample_count"] = self.sample_count
        return data

    def save(self, path: str) -> None:
        self.validate()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        # Counts are derived and are verified rather than trusted on load.
        expected_eligible_count = data.pop("eligible_count", None)
        expected_sample_count = data.pop("sample_count", None)
        manifest = cls(**data)
        manifest.validate()
        if (
            expected_eligible_count is not None
            and expected_eligible_count != manifest.eligible_count
        ):
            raise ValueError("eligible_count does not match eligible_indices.")
        if (
            expected_sample_count is not None
            and expected_sample_count != manifest.sample_count
        ):
            raise ValueError("sample_count does not match selected_indices.")
        return manifest


def jointly_correct_indices(
    labels: Sequence[int],
    new_predictions: Sequence[int],
    old_predictions: Sequence[int],
) -> List[int]:
    """Return indices where both model predictions equal the gold label."""

    if not (
        len(labels) == len(new_predictions) == len(old_predictions)
    ):
        raise ValueError("Labels and prediction sequences must have equal lengths.")
    return [
        index
        for index, (label, new_prediction, old_prediction) in enumerate(
            zip(labels, new_predictions, old_predictions)
        )
        if int(new_prediction) == int(label) and int(old_prediction) == int(label)
    ]


def select_manifest_indices(
    eligible_indices: Iterable[int],
    *,
    strategy: str,
    sample_size: Optional[int] = None,
    seed: int = 765,
) -> List[int]:
    """Select eligible indices using an explicit dataset-agnostic policy."""

    eligible = sorted(eligible_indices)
    if not eligible:
        raise ValueError("No jointly correct eligible samples are available.")
    if strategy == "all":
        return eligible
    if strategy not in {"random_exact", "random_up_to"}:
        raise ValueError(
            "strategy must be 'all', 'random_exact', or 'random_up_to'."
        )
    if sample_size is None or sample_size <= 0:
        raise ValueError("A positive sample_size is required for random sampling.")
    if strategy == "random_exact" and len(eligible) < sample_size:
        raise ValueError(
            f"Requested {sample_size} eligible samples, got {len(eligible)}."
        )
    selected_size = min(sample_size, len(eligible))
    # Sorting before sampling makes the result independent of iterator order.
    return random.Random(seed).sample(eligible, selected_size)


class ManifestDatasetView(Dataset):
    """Read-only dataset view whose order is fixed by a sample manifest."""

    def __init__(self, dataset: Dataset, manifest: SampleManifest):
        manifest.validate()
        if any(index >= len(dataset) for index in manifest.selected_indices):
            raise ValueError("Manifest selects an index outside the loaded dataset.")
        self._source_dataset = dataset
        self._selected_indices = tuple(manifest.selected_indices)
        self.manifest = manifest
        self.input_columns = dataset.input_columns
        self.label_names = dataset.label_names
        self.label_map = getattr(dataset, "label_map", None)
        self.output_scale_factor = getattr(dataset, "output_scale_factor", None)
        self.shuffled = False

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._source_dataset[self._selected_indices[index]]
        indices = self._selected_indices[index]
        return [self._source_dataset[source_index] for source_index in indices]

    def __len__(self):
        return len(self._selected_indices)

    def source_index(self, view_index: int) -> int:
        """Map a run-local position back to the frozen dataset index."""

        return self._selected_indices[view_index]
