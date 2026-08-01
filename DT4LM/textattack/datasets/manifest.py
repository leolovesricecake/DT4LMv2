"""Immutable, model-independent sample manifests for DT4LM experiments."""

from dataclasses import asdict, dataclass
import json
from typing import List, Optional

from ..dt4lm_sampling import (
    SAMPLING_ALGORITHM_ALL,
    SAMPLING_ALGORITHM_HASH,
    select_sample_indices,
    selection_hash,
)

from .dataset import Dataset


@dataclass(frozen=True)
class SampleManifest:
    """The exact ordered random sample selected from one dataset split."""

    schema_version: int
    dataset_id: str
    dataset_fingerprint: str
    dataset_revision: Optional[str]
    split: str
    population_size: int
    requested_sample_size: Optional[int]
    effective_sample_size: int
    seed: int
    sampling_algorithm: str
    selected_indices: List[int]
    selection_sha256: str

    @property
    def sample_count(self) -> int:
        """Retain the conventional name used by TextAttack metric adapters."""

        return self.effective_sample_size

    def validate(self) -> None:
        """Reject malformed manifests before they can alter metric denominators."""

        if self.schema_version != 2:
            raise ValueError("Sample manifest schema_version must be 2.")
        if not self.dataset_id or not self.split:
            raise ValueError("Manifest dataset_id and split must be non-empty.")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive.")
        if self.effective_sample_size != len(self.selected_indices):
            raise ValueError("effective_sample_size does not match selected_indices.")
        if not self.selected_indices:
            raise ValueError("A sample manifest must select at least one row.")
        if len(set(self.selected_indices)) != len(self.selected_indices):
            raise ValueError("selected_indices contains duplicates.")
        if any(
            index < 0 or index >= self.population_size
            for index in self.selected_indices
        ):
            raise ValueError("Manifest indices must lie inside the source split.")
        if self.sampling_algorithm not in {
            SAMPLING_ALGORITHM_ALL,
            SAMPLING_ALGORITHM_HASH,
        }:
            raise ValueError("Unsupported sampling_algorithm in sample manifest.")
        expected_hash = selection_hash(
            self.dataset_id,
            self.dataset_fingerprint,
            self.split,
            self.seed,
            self.selected_indices,
        )
        if self.selection_sha256 != expected_hash:
            raise ValueError("selection_sha256 does not match selected_indices.")

    def to_dict(self):
        """Return the versioned JSON representation without derived legacy fields."""

        return asdict(self)

    def save(self, path: str) -> None:
        self.validate()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                self.to_dict(), handle, ensure_ascii=True, indent=2, sort_keys=True
            )

    @classmethod
    def load(cls, path: str):
        with open(path, encoding="utf-8") as handle:
            manifest = cls(**json.load(handle))
        manifest.validate()
        return manifest

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        dataset_fingerprint: str,
        dataset_revision: Optional[str],
        split: str,
        population_size: int,
        requested_sample_size: Optional[int],
        seed: int,
    ):
        """Build a validated manifest from the public sampling protocol."""

        selected = select_sample_indices(
            population_size,
            sample_size=requested_sample_size,
            seed=seed,
            dataset_fingerprint=dataset_fingerprint,
            split=split,
        )
        algorithm = (
            SAMPLING_ALGORITHM_ALL
            if len(selected) == population_size
            else SAMPLING_ALGORITHM_HASH
        )
        manifest = cls(
            schema_version=2,
            dataset_id=dataset_id,
            dataset_fingerprint=dataset_fingerprint,
            dataset_revision=dataset_revision,
            split=split,
            population_size=population_size,
            requested_sample_size=requested_sample_size,
            effective_sample_size=len(selected),
            seed=int(seed),
            sampling_algorithm=algorithm,
            selected_indices=selected,
            selection_sha256=selection_hash(
                dataset_id, dataset_fingerprint, split, seed, selected
            ),
        )
        manifest.validate()
        return manifest


class ManifestDatasetView(Dataset):
    """Read-only dataset view whose order is fixed by a sample manifest."""

    def __init__(self, dataset: Dataset, manifest: SampleManifest):
        manifest.validate()
        source = getattr(dataset, "_dataset", None)
        actual_fingerprint = getattr(source, "_fingerprint", None)
        if (
            manifest.dataset_fingerprint
            and actual_fingerprint
            and manifest.dataset_fingerprint != actual_fingerprint
        ):
            raise ValueError(
                "Loaded dataset fingerprint does not match the frozen manifest: "
                f"{actual_fingerprint!r} != {manifest.dataset_fingerprint!r}."
            )
        if len(dataset) != manifest.population_size:
            raise ValueError(
                "Loaded dataset size does not match the frozen manifest: "
                f"{len(dataset)} != {manifest.population_size}."
            )
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
