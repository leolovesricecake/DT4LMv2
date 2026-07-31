"""Shared identity and path rules for reproducible DT4LM artifacts."""

import json
from pathlib import Path
import re


_ARTIFACT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def require_model_pair_id(config):
    """Return a filesystem-safe ``models.id`` or fail with a clear message."""

    models = config.get("models")
    if not isinstance(models, dict):
        raise ValueError("Dataset configuration requires mapping section 'models'.")
    model_pair_id = models.get("id")
    if not isinstance(model_pair_id, str) or not _ARTIFACT_ID.fullmatch(
        model_pair_id
    ):
        raise ValueError(
            "models.id must be a non-empty filesystem-safe identifier using only "
            "letters, digits, '.', '_' and '-'."
        )
    if model_pair_id in {".", ".."}:
        raise ValueError("models.id cannot be '.' or '..'.")
    return model_pair_id


def artifact_namespace(config):
    """Return the dataset/model-pair namespace shared by all artifacts."""

    dataset = config.get("dataset")
    if not isinstance(dataset, dict) or not dataset.get("id"):
        raise ValueError("Dataset configuration requires dataset.id.")
    dataset_id = str(dataset["id"])
    if not _ARTIFACT_ID.fullmatch(dataset_id) or dataset_id in {".", ".."}:
        raise ValueError(
            "dataset.id must be a filesystem-safe identifier using only letters, "
            "digits, '.', '_' and '-'."
        )
    return dataset_id, require_model_pair_id(config)


def resolve_path(project_root, configured):
    """Resolve a configured artifact path relative to the DT4LM root."""

    path = Path(configured).expanduser()
    return path if path.is_absolute() else Path(project_root) / path


def resolve_model_id(project_root, configured):
    """Resolve local checkpoints while preserving Hugging Face model IDs."""

    path = Path(configured).expanduser()
    candidate = path if path.is_absolute() else Path(project_root) / path
    return str(candidate.resolve()) if candidate.exists() else str(configured)


def _require_namespace_suffix(path, dataset_id, model_pair_id, label, file_path):
    """Require paths to end in the configured dataset/model-pair namespace."""

    namespace_path = path.parent if file_path else path
    expected = (dataset_id, model_pair_id)
    if tuple(namespace_path.parts[-2:]) != expected:
        expected_text = f"<root>/{dataset_id}/{model_pair_id}"
        if file_path:
            expected_text += f"/{path.name}"
        raise ValueError(
            f"{label} must be namespaced as {expected_text}; got {path}."
        )


def validate_artifact_namespaces(config, project_root):
    """Validate manifest and calibration paths against dataset and model pair."""

    dataset_id, model_pair_id = artifact_namespace(config)
    manifests = config.get("manifests")
    if not isinstance(manifests, dict):
        raise ValueError("Dataset configuration requires mapping section 'manifests'.")
    for role in ("train", "test", "metadata"):
        configured = manifests.get(role)
        if not configured:
            raise ValueError(f"manifests is missing required field {role!r}.")
        _require_namespace_suffix(
            resolve_path(project_root, configured),
            dataset_id,
            model_pair_id,
            f"manifests.{role}",
            file_path=True,
        )

    calibration = config.get("calibration")
    if calibration is not None:
        if not isinstance(calibration, dict) or not calibration.get("output_root"):
            raise ValueError(
                "calibration.output_root is required when calibration exists."
            )
        _require_namespace_suffix(
            resolve_path(project_root, calibration["output_root"]),
            dataset_id,
            model_pair_id,
            "calibration.output_root",
            file_path=False,
        )


def run_directory(config, project_root, experiment_name):
    """Build the collision-free directory for one formal experiment run."""

    dataset_id, model_pair_id = artifact_namespace(config)
    output_root = resolve_path(project_root, config["output_root"])
    return output_root / "runs" / dataset_id / model_pair_id / experiment_name


def validate_manifest_identity(manifest_path, config, expected_split, project_root):
    """Fail before model loading when a manifest belongs to another experiment."""

    path = Path(manifest_path)
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    dataset_id, model_pair_id = artifact_namespace(config)
    expected = {
        "dataset_id": dataset_id,
        "model_pair_id": model_pair_id,
        "split": expected_split,
        "old_model_id": resolve_model_id(project_root, config["models"]["old"]),
        "new_model_id": resolve_model_id(project_root, config["models"]["new"]),
    }
    mismatches = {
        key: {"manifest": manifest.get(key), "configured": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Manifest {path} does not match the configured dataset/model pair: "
            f"{mismatches!r}. Regenerate it in the correct namespace."
        )
    return manifest
