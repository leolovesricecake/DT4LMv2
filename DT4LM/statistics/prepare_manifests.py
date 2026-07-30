#!/usr/bin/env python
"""Freeze jointly-correct train and test manifests for first-round runs."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import yaml


def _load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for section in ("dataset", "models", "manifests", "sampling"):
        if section not in config:
            raise ValueError(f"Experiment config is missing {section!r}.")
    return config


def _model_revision(model, configured):
    """Prefer a resolved Hub commit while retaining local/checkpoint revisions."""

    return (
        getattr(model.config, "_commit_hash", None)
        or configured
        or getattr(model.config, "_name_or_path", None)
    )


def _resolve_checkpoint(project_root, configured):
    """Resolve tracked local checkpoint paths while preserving Hub model IDs."""

    candidate = project_root / configured
    return str(candidate) if candidate.exists() else configured


def _predict(dataset, model_id, revision, columns, label_column, batch_size, device):
    """Predict one complete split with its checkpoint-specific tokenizer."""

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision
    )
    model.to(device)
    model.eval()
    predictions = []
    labels = []
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            rows = dataset[start : start + batch_size]
            texts = [rows[column] for column in columns]
            encoded = tokenizer(
                *texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            predictions.extend(model(**encoded).logits.argmax(dim=-1).cpu().tolist())
            labels.extend(int(value) for value in rows[label_column])
    return labels, predictions, model


def _manifest(
    *,
    dataset_id,
    fingerprint,
    split,
    old_model_id,
    new_model_id,
    old_revision,
    new_revision,
    seed,
    population_size,
    eligible,
    selected,
):
    """Build the exact schema consumed by ManifestDatasetView and statistics."""

    return {
        "dataset_id": dataset_id,
        "dataset_revision_or_fingerprint": fingerprint,
        "split": split,
        "old_model_id": old_model_id,
        "new_model_id": new_model_id,
        "old_model_revision": old_revision,
        "new_model_revision": new_revision,
        "seed": seed,
        "test_split_size": population_size,
        "eligible_indices": list(eligible),
        "selected_indices": list(selected),
        "eligible_count": len(eligible),
        "sample_count": len(selected),
    }


def _jointly_correct(labels, new_predictions, old_predictions):
    if not (len(labels) == len(new_predictions) == len(old_predictions)):
        raise ValueError("Prediction and label lengths differ.")
    return [
        index
        for index, values in enumerate(
            zip(labels, new_predictions, old_predictions)
        )
        if values[0] == values[1] == values[2]
    ]


def _select_indices(eligible, policy, seed, role):
    """Apply a generic, config-defined manifest sampling policy."""

    if not isinstance(policy, dict):
        raise ValueError(f"sampling.{role} must be a YAML mapping.")
    strategy = policy.get("strategy")
    population = sorted(eligible)
    if not population:
        raise ValueError(f"No jointly-correct samples are available for {role}.")
    if strategy == "all":
        return population
    if strategy not in {"random_exact", "random_up_to"}:
        raise ValueError(
            f"sampling.{role}.strategy must be all, random_exact, or random_up_to."
        )
    size = policy.get("size")
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"sampling.{role}.size must be a positive integer.")
    if strategy == "random_exact" and len(population) < size:
        raise ValueError(
            f"sampling.{role} requested exactly {size} samples, "
            f"but only {len(population)} are eligible."
        )
    # random_up_to preserves a useful small-dataset fallback without hiding the
    # requested cap in Python code.
    selected_size = min(size, len(population))
    return random.Random(seed).sample(population, selected_size)


def _resolve_output(project_root, configured):
    """Resolve one configured manifest artifact path from the DT4LM root."""

    path = Path(configured)
    return path if path.is_absolute() else project_root / path


def _load_dataset_collection(dataset_config, project_root):
    """Load the same local-or-Hub DatasetDict consumed by TextAttack."""

    from datasets import load_dataset, load_from_disk

    configured = Path(dataset_config["path"]).expanduser()
    local_path = (
        configured
        if configured.is_absolute()
        else project_root / configured
    )
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    config = _load_config(args.config)

    import torch
    dataset_config = config["dataset"]
    model_config = config["models"]
    project_root = Path(__file__).resolve().parents[1]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaded = _load_dataset_collection(dataset_config, project_root)
    columns = list(dataset_config["text_columns"])
    label_column = dataset_config.get("label_column", "label")
    old_id = _resolve_checkpoint(project_root, model_config["old"])
    new_id = _resolve_checkpoint(project_root, model_config["new"])
    old_configured_revision = model_config.get("old_revision")
    new_configured_revision = model_config.get("new_revision")
    seed = int(config.get("seed", 765))
    manifests = {}

    for role, split_key, sampling_key in (
        ("test", "test_split", "test"),
        ("train", "calibration_split", "calibration_originals"),
    ):
        split = dataset_config[split_key]
        dataset = loaded[split]
        labels, new_predictions, new_model = _predict(
            dataset,
            new_id,
            new_configured_revision,
            columns,
            label_column,
            args.batch_size,
            device,
        )
        old_labels, old_predictions, old_model = _predict(
            dataset,
            old_id,
            old_configured_revision,
            columns,
            label_column,
            args.batch_size,
            device,
        )
        if labels != old_labels:
            raise ValueError("Model passes observed inconsistent dataset labels.")
        if new_model.config.num_labels != old_model.config.num_labels:
            raise ValueError("Old and new checkpoints have different label counts.")
        new_mapping = {
            int(index): str(value).lower()
            for index, value in new_model.config.id2label.items()
        }
        old_mapping = {
            int(index): str(value).lower()
            for index, value in old_model.config.id2label.items()
        }
        if new_mapping != old_mapping:
            raise ValueError("Old and new checkpoints have different label mappings.")
        eligible = _jointly_correct(labels, new_predictions, old_predictions)
        selected = _select_indices(
            eligible,
            config["sampling"].get(sampling_key),
            seed,
            sampling_key,
        )
        fingerprint = getattr(dataset, "_fingerprint", "")
        manifests[role] = _manifest(
            dataset_id=dataset_config["id"],
            fingerprint=fingerprint,
            split=split,
            old_model_id=old_id,
            new_model_id=new_id,
            old_revision=_model_revision(old_model, old_configured_revision),
            new_revision=_model_revision(new_model, new_configured_revision),
            seed=seed,
            population_size=len(dataset),
            eligible=eligible,
            selected=selected,
        )
        # Releasing each pair prevents both split passes from retaining models.
        del old_model, new_model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    for role, manifest in manifests.items():
        output_path = _resolve_output(project_root, config["manifests"][role])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=True, indent=2, sort_keys=True)
    with open(args.config, "rb") as handle:
        config_hash = hashlib.sha256(handle.read()).hexdigest()
    metadata_path = _resolve_output(
        project_root,
        config["manifests"].get(
            "metadata",
            str(Path(config["manifests"]["test"]).parent / "manifest_metadata.json"),
        ),
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config_sha256": config_hash,
                "device": device,
                "sampling": config["sampling"],
                "selected_counts": {
                    role: len(manifest["selected_indices"])
                    for role, manifest in manifests.items()
                },
            },
            handle,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
