#!/usr/bin/env python
"""Convert the adversarial-training notebook into restartable local CLI stages."""

import argparse
import json
from pathlib import Path
import random


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_source(source, split=None):
    """Load a Hub or save_to_disk dataset and optionally select one split."""

    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    local_path = Path(source).expanduser()
    loaded = (
        load_from_disk(str(local_path.resolve()))
        if local_path.exists()
        else load_dataset(source)
    )
    if split is None:
        return loaded
    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            raise KeyError(f"Dataset {source!r} has no split {split!r}.")
        return loaded[split]
    if isinstance(loaded, Dataset):
        if split != "train":
            raise KeyError(
                f"Single local Dataset {source!r} can only be used as train."
            )
        return loaded
    raise TypeError(f"Unsupported dataset object loaded from {source!r}.")


def _save_dataset(dataset, output_dir, metadata):
    """Save without overwriting and attach reproducibility metadata."""

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Choose a new --output-dir."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    with open(
        output_dir / "preprocessing_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2, sort_keys=True)


def sample_training_instances(source, fraction, seed):
    """Select the notebook's fixed random fraction from the train split."""

    from datasets import DatasetDict

    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1].")
    train = _load_source(source, split="train")
    sample_size = int(fraction * len(train))
    if sample_size <= 0:
        raise ValueError(
            f"fraction {fraction} selects no rows from {len(train)} examples."
        )
    indices = random.Random(seed).sample(range(len(train)), sample_size)
    return DatasetDict({"train": train.select(indices)}), indices


def combine_training_data(
    adversarial_source,
    original_source,
    *,
    adversarial_text_column,
    target_text_column,
    seed,
):
    """Append generated examples to train and preserve original eval splits."""

    from datasets import DatasetDict, concatenate_datasets

    adversarial = _load_source(adversarial_source, split="train")
    original = _load_source(original_source)
    if "train" not in original:
        raise ValueError("The original dataset has no train split.")
    if (
        target_text_column not in adversarial.column_names
        and adversarial_text_column in adversarial.column_names
    ):
        adversarial = adversarial.rename_column(
            adversarial_text_column, target_text_column
        )
    missing = set(original["train"].column_names) - set(
        adversarial.column_names
    )
    if missing:
        raise ValueError(
            f"Adversarial data is missing original columns: {sorted(missing)}."
        )
    # Drop attack metadata before cast; this remains compatible with older
    # datasets releases that predate Dataset.select_columns.
    extra_columns = [
        column
        for column in adversarial.column_names
        if column not in original["train"].column_names
    ]
    if extra_columns:
        adversarial = adversarial.remove_columns(extra_columns)
    adversarial = adversarial.cast(original["train"].features)
    combined_train = concatenate_datasets(
        [adversarial, original["train"]]
    ).shuffle(seed=seed)
    return DatasetDict(
        {
            "train": combined_train,
            **{
                split_name: split
                for split_name, split in original.items()
                if split_name != "train"
            },
        }
    )


def _output_path(configured):
    """Resolve the configured output path independently of the caller's cwd."""

    return Path(configured).expanduser().resolve()


def build_parser():
    """Create subcommands matching the two notebook sections."""

    parser = argparse.ArgumentParser(
        description="Prepare local datasets for DT4LM adversarial training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    sample = subparsers.add_parser(
        "sample",
        help="Sample original train rows for differential generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sample.add_argument(
        "--source",
        required=True,
        help="Original local DatasetDict directory or Hub dataset ID.",
    )
    sample.add_argument(
        "--output-dir",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "datasets"
            / "adversarial-training-sample"
        ),
        help="Local DatasetDict output directory.",
    )
    sample.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction of original train examples to sample.",
    )
    sample.add_argument(
        "--seed", type=int, default=42, help="Sampling seed."
    )
    sample.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        help="Optionally upload after saving locally.",
    )
    sample.add_argument(
        "--private",
        action="store_true",
        help="Make the optional Hub repository private.",
    )

    combine = subparsers.add_parser(
        "combine",
        help="Mix generated differential examples with original training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    combine.add_argument(
        "--adversarial-source",
        required=True,
        help="Generated differential dataset directory or Hub ID.",
    )
    combine.add_argument(
        "--original-source",
        required=True,
        help="Original DatasetDict directory or Hub ID.",
    )
    combine.add_argument(
        "--output-dir",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "datasets"
            / "adversarial-training-combined"
        ),
        help="Local combined DatasetDict output directory.",
    )
    combine.add_argument(
        "--adversarial-text-column",
        default="text",
        help="Generated text column before schema alignment.",
    )
    combine.add_argument(
        "--target-text-column",
        default="sentence",
        help="Original dataset text column.",
    )
    combine.add_argument(
        "--seed", type=int, default=42, help="Combined train shuffle seed."
    )
    combine.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        help="Optionally upload after saving locally.",
    )
    combine.add_argument(
        "--private",
        action="store_true",
        help="Make the optional Hub repository private.",
    )
    return parser


def main():
    """Execute one notebook-derived adversarial data stage."""

    args = build_parser().parse_args()
    if args.stage == "sample":
        dataset, indices = sample_training_instances(
            args.source, args.fraction, args.seed
        )
        output_dir = _output_path(args.output_dir)
        metadata = {
            "stage": "sample",
            "source": args.source,
            "fraction": args.fraction,
            "seed": args.seed,
            "sample_size": len(indices),
            "sampled_indices": indices,
        }
    else:
        dataset = combine_training_data(
            args.adversarial_source,
            args.original_source,
            adversarial_text_column=args.adversarial_text_column,
            target_text_column=args.target_text_column,
            seed=args.seed,
        )
        output_dir = _output_path(args.output_dir)
        metadata = {
            "stage": "combine",
            "adversarial_source": args.adversarial_source,
            "original_source": args.original_source,
            "adversarial_text_column": args.adversarial_text_column,
            "target_text_column": args.target_text_column,
            "seed": args.seed,
            "split_sizes": {
                split_name: len(split)
                for split_name, split in dataset.items()
            },
        }
    _save_dataset(dataset, output_dir, metadata)
    if args.push_to_hub:
        dataset.push_to_hub(args.push_to_hub, private=args.private)
    print(json.dumps({**metadata, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
