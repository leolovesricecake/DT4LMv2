#!/usr/bin/env python
"""Freeze model-independent random samples for one complete experiment config."""

import argparse
import json
from pathlib import Path
import sys


# Direct execution adds statistics/ to sys.path, so expose shared pure helpers.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dt4lm_artifacts import resolve_path, validate_artifact_namespaces  # noqa: E402
from dt4lm_sampling import (  # noqa: E402
    SAMPLING_ALGORITHM_ALL,
    SAMPLING_ALGORITHM_HASH,
    select_sample_indices,
    selection_hash,
    validate_sample_manifest_payload,
)
from improvement_config import load_experiment_config  # noqa: E402


def _load_dataset_collection(dataset_config, project_root):
    """Load the local-or-Hub DatasetDict without loading either target model."""

    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "Hugging Face 'datasets' is required to prepare sample manifests. "
            "Install the DT4LM environment before running this command."
        ) from exc

    configured = Path(dataset_config["path"]).expanduser()
    local_path = configured if configured.is_absolute() else project_root / configured
    if local_path.exists():
        return load_from_disk(str(local_path.resolve()))
    explicitly_local = configured.is_absolute() or str(
        dataset_config["path"]
    ).startswith((".", "outputs/"))
    if explicitly_local:
        raise FileNotFoundError(
            f"Local dataset does not exist: {local_path}. "
            "Run datasets/preprocess_dataset.py first."
        )
    return load_dataset(
        dataset_config["path"],
        dataset_config.get("name"),
        revision=dataset_config.get("revision"),
    )


def build_manifest(dataset_config, dataset, sampling):
    """Build the version-2 JSON payload for one split sampling request."""

    split = str(sampling["split"])
    fingerprint = str(getattr(dataset, "_fingerprint", "") or "")
    requested = sampling.get("sample_size")
    seed = int(sampling["sample_seed"])
    selected = select_sample_indices(
        len(dataset),
        sample_size=requested,
        seed=seed,
        dataset_fingerprint=fingerprint,
        split=split,
    )
    return {
        "schema_version": 2,
        "dataset_id": str(dataset_config["id"]),
        "dataset_fingerprint": fingerprint,
        "dataset_revision": dataset_config.get("revision"),
        "split": split,
        "population_size": len(dataset),
        "requested_sample_size": requested,
        "effective_sample_size": len(selected),
        "seed": seed,
        "sampling_algorithm": (
            SAMPLING_ALGORITHM_ALL
            if len(selected) == len(dataset)
            else SAMPLING_ALGORITHM_HASH
        ),
        "selected_indices": selected,
        "selection_sha256": selection_hash(
            str(dataset_config["id"]), fingerprint, split, seed, selected
        ),
    }


def _write_frozen_payloads(payloads):
    """Create immutable JSON artifacts, allowing only identical reuse."""

    # Validate existing siblings before creating any missing file so a conflict
    # cannot leave a partially updated manifest set.
    for path, payload in payloads:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != payload:
            raise FileExistsError(
                f"Refusing to overwrite frozen sample manifest {path}. "
                "Use a new path after changing data or sampling parameters."
            )

    for path, payload in payloads:
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)


def configured_sampling_requests(config):
    """Return test and optional calibration-source sampling definitions."""

    requests = [("test", config["dataset"]["evaluation"])]
    calibration = config["calibration"]
    if calibration["enabled"]:
        requests.append(("calibration", calibration["source_sampling"]))
    return requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_experiment_config(Path(args.config).resolve())
    validate_artifact_namespaces(config, PROJECT_ROOT)
    collection = _load_dataset_collection(config["dataset"], PROJECT_ROOT)
    payloads = []
    for role, sampling in configured_sampling_requests(config):
        split = str(sampling["split"])
        if split not in collection:
            raise ValueError(f"Dataset has no configured {role} split {split!r}.")
        payload = build_manifest(config["dataset"], collection[split], sampling)
        validate_sample_manifest_payload(payload)
        payloads.append(
            (resolve_path(PROJECT_ROOT, sampling["manifest"]), payload)
        )
    _write_frozen_payloads(payloads)


if __name__ == "__main__":
    main()
