"""Pure sampling primitives shared by manifest generation and consumption."""

import hashlib
import json
from typing import List, Optional


SAMPLING_ALGORITHM_ALL = "all_v1"
SAMPLING_ALGORITHM_HASH = "sha256_rank_v1"


def selection_hash(
    dataset_id: str,
    fingerprint: str,
    split: str,
    seed: int,
    indices: List[int],
) -> str:
    """Hash the dataset identity and ordered sample without hashing its file."""

    payload = [dataset_id, fingerprint, split, int(seed), list(indices)]
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_sample_indices(
    population_size: int,
    *,
    sample_size: Optional[int],
    seed: int,
    dataset_fingerprint: str,
    split: str,
) -> List[int]:
    """Select at most ``sample_size`` rows with stable hash-based randomness."""

    if population_size <= 0:
        raise ValueError("population_size must be positive.")
    if sample_size is None or sample_size <= 0 or sample_size >= population_size:
        return list(range(population_size))

    def rank(index: int) -> bytes:
        # Including data identity prevents accidental sample reuse after a split
        # changes while remaining stable across Python and NumPy versions.
        value = f"{dataset_fingerprint}\0{split}\0{int(seed)}\0{index}"
        return hashlib.sha256(value.encode("utf-8")).digest()

    return sorted(range(population_size), key=rank)[:sample_size]


def validate_sample_manifest_payload(payload) -> None:
    """Validate a version-2 manifest without importing TextAttack or models."""

    required = {
        "schema_version",
        "dataset_id",
        "dataset_fingerprint",
        "split",
        "population_size",
        "effective_sample_size",
        "seed",
        "sampling_algorithm",
        "selected_indices",
        "selection_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Sample manifest is missing fields: {missing!r}.")
    if payload["schema_version"] != 2:
        raise ValueError("Sample manifest schema_version must be 2.")
    population_size = int(payload["population_size"])
    indices = list(payload["selected_indices"])
    if population_size <= 0 or not indices:
        raise ValueError("Sample manifest population and selection must be non-empty.")
    if int(payload["effective_sample_size"]) != len(indices):
        raise ValueError("effective_sample_size does not match selected_indices.")
    if len(indices) != len(set(indices)):
        raise ValueError("selected_indices contains duplicates.")
    if any(index < 0 or index >= population_size for index in indices):
        raise ValueError("Sample manifest indices fall outside the source split.")
    if payload["sampling_algorithm"] not in {
        SAMPLING_ALGORITHM_ALL,
        SAMPLING_ALGORITHM_HASH,
    }:
        raise ValueError("Sample manifest uses an unsupported sampling algorithm.")
    expected_hash = selection_hash(
        str(payload["dataset_id"]),
        str(payload["dataset_fingerprint"]),
        str(payload["split"]),
        int(payload["seed"]),
        indices,
    )
    if payload["selection_sha256"] != expected_hash:
        raise ValueError("selection_sha256 does not match selected_indices.")
