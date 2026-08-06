"""Shared identity and path rules for reproducible DT4LM artifacts."""

import json
from pathlib import Path
import re

from dt4lm_sampling import validate_sample_manifest_payload


_ARTIFACT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def require_model_pair_id(config):
    """Return a filesystem-safe ``models.id`` or fail with a clear message."""

    models = config.get("models")
    if not isinstance(models, dict):
        raise ValueError("Experiment configuration requires mapping section 'models'.")
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

    if isinstance(configured, dict):
        configured = configured.get("name_or_path")
    if not configured:
        raise ValueError("A model name_or_path is required.")
    path = Path(configured).expanduser()
    candidate = path if path.is_absolute() else Path(project_root) / path
    return str(candidate.resolve()) if candidate.exists() else str(configured)


def _require_namespace_suffix(path, expected, label, file_path):
    """Require paths to end in a configured artifact namespace."""

    namespace_path = path.parent if file_path else path
    if tuple(namespace_path.parts[-len(expected) :]) != tuple(expected):
        expected_text = "<root>/" + "/".join(expected)
        if file_path:
            expected_text += f"/{path.name}"
        raise ValueError(
            f"{label} must be namespaced as {expected_text}; got {path}."
        )


def validate_artifact_namespaces(config, project_root):
    """Validate model-independent samples and model-specific run artifacts."""

    dataset_id, model_pair_id = artifact_namespace(config)
    models = config.get("models") or {}
    old_spec = models.get("old") or {}
    new_spec = models.get("new") or {}
    old_model = resolve_model_id(project_root, old_spec)
    new_model = resolve_model_id(project_root, new_spec)
    same_revision = old_spec.get("revision") == new_spec.get("revision")
    same_local_checkpoint = old_model == new_model and Path(old_model).is_absolute()
    if old_model == new_model and (same_revision or same_local_checkpoint):
        raise ValueError(
            "models.old and models.new resolve to the same checkpoint; "
            "refusing to run a degenerate differential experiment."
        )
    dataset = config.get("dataset") or {}
    evaluation = dataset.get("evaluation") or {}
    configured = evaluation.get("manifest")
    if not configured:
        raise ValueError("dataset.evaluation.manifest is required.")
    _require_namespace_suffix(
        resolve_path(project_root, configured),
        (dataset_id,),
        "dataset.evaluation.manifest",
        file_path=True,
    )

    calibration = config.get("calibration")
    if isinstance(calibration, dict) and calibration.get("enabled"):
        source = calibration.get("source_sampling") or {}
        source_manifest = source.get("manifest")
        if not source_manifest:
            raise ValueError("calibration.source_sampling.manifest is required.")
        _require_namespace_suffix(
            resolve_path(project_root, source_manifest),
            (dataset_id,),
            "calibration.source_sampling.manifest",
            file_path=True,
        )
        backend = str((calibration.get("judge") or {}).get("backend") or "")
        if not calibration.get("output_dir") or not backend:
            raise ValueError("Enabled calibration requires output_dir and judge.backend.")
        _require_namespace_suffix(
            resolve_path(project_root, calibration["output_dir"]),
            (dataset_id, model_pair_id, backend),
            "calibration.output_dir",
            file_path=False,
        )


def run_directory(config, project_root, experiment_name=None):
    """Build the collision-free directory for one formal experiment run."""

    dataset_id, model_pair_id = artifact_namespace(config)
    experiment = config.get("experiment") or {}
    output_root = resolve_path(project_root, experiment["output_root"])
    experiment_name = experiment_name or experiment["id"]
    return output_root / "runs" / dataset_id / model_pair_id / experiment_name


def validate_manifest_identity(manifest_path, config, expected_split, project_root):
    """Fail before model loading when a sample manifest belongs to other data."""

    path = Path(manifest_path)
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_sample_manifest_payload(manifest)
    dataset_id, _ = artifact_namespace(config)
    expected = {
        "dataset_id": dataset_id,
        "split": expected_split,
    }
    mismatches = {
        key: {"manifest": manifest.get(key), "configured": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Manifest {path} does not match the configured dataset split: "
            f"{mismatches!r}. Regenerate it in the correct dataset namespace."
        )
    return manifest
