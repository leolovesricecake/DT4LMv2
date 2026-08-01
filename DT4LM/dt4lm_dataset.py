"""Shared loading and schema checks for DT4LM experiment datasets."""

from pathlib import Path


def load_dataset_collection(dataset_config, project_root):
    """Load one local or Hub dataset collection without loading target models."""

    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "Hugging Face 'datasets' is required for DT4LM dataset preflight."
        ) from exc

    configured = Path(dataset_config["path"]).expanduser()
    local_path = (
        configured if configured.is_absolute() else Path(project_root) / configured
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


def validate_dataset_split_schema(dataset_config, split_dataset, split):
    """Require configured text and label columns to match a concrete split."""

    actual = list(getattr(split_dataset, "column_names", ()) or ())
    expected = [
        *dataset_config["text_columns"],
        dataset_config["label_column"],
    ]
    if actual != expected:
        raise ValueError(
            f"Dataset {dataset_config['id']!r} split {split!r} schema mismatch: "
            f"configured columns are {expected!r}, but the dataset contains "
            f"{actual!r}. Reprocess the dataset or correct dataset.text_columns/"
            "label_column before preparing manifests or loading models."
        )
    return tuple(actual)
