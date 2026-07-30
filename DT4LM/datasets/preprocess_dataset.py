#!/usr/bin/env python
"""Convert the repository's supervised-data notebooks into one reproducible CLI."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_TEMPLATE = "outputs/datasets/<dataset>"


@dataclass(frozen=True)
class DatasetSpec:
    """Notebook-derived source and transformation settings for one task."""

    source: str
    subset: str
    valid_labels: tuple
    rename_columns: tuple = ()
    shuffle_complete_splits: bool = False


# These definitions reproduce the four fine-tuning preprocessing notebooks.
DATASET_SPECS = {
    "sst2": DatasetSpec(
        source="nyu-mll/glue",
        subset="sst2",
        valid_labels=(0, 1),
    ),
    "rte": DatasetSpec(
        source="nyu-mll/glue",
        subset="rte",
        valid_labels=(0, 1),
        rename_columns=(
            ("sentence1", "premise"),
            ("sentence2", "hypothesis"),
        ),
    ),
    "mrpc": DatasetSpec(
        source="nyu-mll/glue",
        subset="mrpc",
        valid_labels=(0, 1),
    ),
    "mr": DatasetSpec(
        source="cornell-movie-review-data/rotten_tomatoes",
        subset=None,
        valid_labels=(0, 1),
        shuffle_complete_splits=True,
    ),
}


def default_output_dir(dataset_name):
    """Return a stable local output path independent of the caller's cwd."""

    return PROJECT_ROOT / "outputs" / "datasets" / dataset_name


def _load_source(source, subset=None, revision=None):
    """Load either a Hub dataset or a previously saved local DatasetDict."""

    from datasets import load_dataset, load_from_disk

    local_path = Path(source).expanduser()
    if local_path.exists():
        return load_from_disk(str(local_path.resolve()))
    load_kwargs = {}
    if revision:
        load_kwargs["revision"] = revision
    return load_dataset(source, subset, **load_kwargs)


def _filter_labels(split, valid_labels):
    """Discard GLUE test placeholders and any other unsupported labels."""

    valid_labels = set(valid_labels)
    return split.filter(lambda row: row["label"] in valid_labels)


def _has_valid_labels(split, valid_labels):
    """Check a split without materializing another filtered dataset."""

    valid_labels = set(valid_labels)
    return any(label in valid_labels for label in split["label"])


def _create_stratified_splits(
    dataset,
    *,
    test_size,
    validation_size,
    seed,
):
    """Reproduce the notebooks' two-stage sklearn stratified 8:1:1 split."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must lie strictly between zero and one.")
    if not 0 < validation_size < 1 - test_size:
        raise ValueError(
            "validation_size must be positive and leave a non-empty train split."
        )

    from datasets import Dataset, DatasetDict
    from sklearn.model_selection import train_test_split

    frame = dataset.to_pandas()
    train_validation, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame["label"],
    )
    relative_validation_size = validation_size / (1 - test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation_size,
        random_state=seed,
        stratify=train_validation["label"],
    )

    # preserve_index=False avoids the notebook's accidental __index_level_0__.
    result = DatasetDict(
        {
            "train": Dataset.from_pandas(
                train.reset_index(drop=True), preserve_index=False
            ),
            "validation": Dataset.from_pandas(
                validation.reset_index(drop=True), preserve_index=False
            ),
            "test": Dataset.from_pandas(
                test.reset_index(drop=True), preserve_index=False
            ),
        }
    )
    label_feature = dataset.features.get("label")
    if getattr(label_feature, "names", None):
        # Dataset.from_pandas loses ClassLabel names unless they are restored.
        for split_name in result:
            result[split_name] = result[split_name].cast_column(
                "label", label_feature
            )
    return result


def _remove_technical_columns(dataset):
    """Remove source and pandas row IDs that are not model inputs."""

    for split_name in dataset:
        removable = [
            column
            for column in ("idx", "__index_level_0__")
            if column in dataset[split_name].column_names
        ]
        if removable:
            dataset[split_name] = dataset[split_name].remove_columns(removable)
    return dataset


def _rename_columns(dataset, mappings):
    """Apply task-level input names consistently to every split."""

    for source, target in mappings:
        for split_name in dataset:
            columns = dataset[split_name].column_names
            if source not in columns:
                raise ValueError(
                    f"Cannot rename missing column {source!r} in split "
                    f"{split_name!r}; columns are {columns!r}."
                )
            dataset[split_name] = dataset[split_name].rename_column(
                source, target
            )
    return dataset


def preprocess_dataset(
    source_dataset,
    spec,
    *,
    test_size=0.1,
    validation_size=0.1,
    seed=42,
):
    """Apply the selected notebook's filtering, splitting, and schema rules."""

    from datasets import DatasetDict, concatenate_datasets

    available = set(source_dataset.keys())
    if "train" not in available:
        raise ValueError("The source dataset has no train split.")

    complete = {"train", "validation", "test"} <= available
    if complete and _has_valid_labels(
        source_dataset["test"], spec.valid_labels
    ):
        processed = DatasetDict(
            {
                name: _filter_labels(
                    source_dataset[name], spec.valid_labels
                )
                for name in ("train", "validation", "test")
            }
        )
        if spec.shuffle_complete_splits:
            processed = DatasetDict(
                {
                    name: split.shuffle(seed=seed)
                    for name, split in processed.items()
                }
            )
    else:
        # GLUE SST-2/RTE/MRPC test labels are placeholders, so the notebooks
        # combine every available labeled source split before making 8:1:1.
        source_splits = [
            source_dataset[name]
            for name in ("train", "validation", "test")
            if name in available
        ]
        combined = _filter_labels(
            concatenate_datasets(source_splits), spec.valid_labels
        )
        processed = _create_stratified_splits(
            combined,
            test_size=test_size,
            validation_size=validation_size,
            seed=seed,
        )

    processed = _remove_technical_columns(processed)
    processed = _rename_columns(processed, spec.rename_columns)
    for split_name, split in processed.items():
        if not len(split):
            raise ValueError(f"Processed split {split_name!r} is empty.")
    return DatasetDict(processed)


def _write_dataset(dataset, output_dir, metadata):
    """Write an immutable local DatasetDict plus human-readable provenance."""

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Choose a new --output-dir to avoid overwriting an experiment."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    with open(
        output_dir / "preprocessing_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2, sort_keys=True)


def build_parser():
    """Build the CLI without importing heavyweight dataset dependencies."""

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess SST-2, RTE, MRPC, or MR using the original notebook "
            "logic and save a local Hugging Face DatasetDict."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", choices=sorted(DATASET_SPECS))
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_TEMPLATE,
        help="Local DatasetDict output directory.",
    )
    parser.add_argument("--source", help="Override the notebook's source ID.")
    parser.add_argument("--subset", help="Override the source subset/config.")
    parser.add_argument("--revision", help="Optional source dataset revision.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by split and shuffle operations.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Final test fraction when labeled source splits are recombined.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.1,
        help="Final validation fraction when source splits are recombined.",
    )
    parser.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        help="Optionally upload the processed DatasetDict after saving locally.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the optional Hub repository private.",
    )
    return parser


def main():
    """Load, preprocess, save, and summarize one selected dataset."""

    args = build_parser().parse_args()
    spec = DATASET_SPECS[args.dataset]
    source = args.source or spec.source
    subset = args.subset if args.subset is not None else spec.subset
    output_dir = (
        default_output_dir(args.dataset)
        if args.output_dir == DEFAULT_OUTPUT_TEMPLATE
        else Path(args.output_dir).expanduser().resolve()
    )
    source_dataset = _load_source(source, subset, args.revision)
    processed = preprocess_dataset(
        source_dataset,
        spec,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    metadata = {
        "dataset": args.dataset,
        "source": source,
        "subset": subset,
        "revision": args.revision,
        "seed": args.seed,
        "test_size": args.test_size,
        "validation_size": args.validation_size,
        "spec": asdict(spec),
        "split_sizes": {
            split_name: len(split)
            for split_name, split in processed.items()
        },
    }
    _write_dataset(processed, output_dir, metadata)
    if args.push_to_hub:
        processed.push_to_hub(args.push_to_hub, private=args.private)
    print(json.dumps({**metadata, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
