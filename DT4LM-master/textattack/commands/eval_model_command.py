"""

EvalModelCommand class
==============================

"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass

import scipy
import torch
import numpy as np
from datasets import Dataset, load_dataset

import textattack
from textattack import DatasetArgs, ModelArgs
from textattack.commands import TextAttackCommand
from textattack.model_args import HUGGINGFACE_MODELS, TEXTATTACK_MODELS

logger = textattack.shared.utils.logger


def _cb(s):
    return textattack.shared.utils.color_text(str(s), color="blue", method="ansi")


@dataclass
class ModelEvalArgs(ModelArgs, DatasetArgs):
    random_seed: int = 765
    batch_size: int = 32
    num_examples: int = 5
    num_examples_offset: int = 0


class EvalModelCommand(TextAttackCommand):
    """The TextAttack model benchmarking module:

    A command line parser to evaluatate a model from user
    specifications.
    """

    def get_preds(self, model, inputs):
        with torch.no_grad():
            preds = textattack.shared.utils.batch_model_predict(model, inputs)
        return preds

    def test_model_on_dataset(self, args):
        model = ModelArgs._create_model_from_args(args)
        dataset = DatasetArgs._create_dataset_from_args(args)
        if args.num_examples == -1:
            args.num_examples = len(dataset)

        preds = []
        ground_truth_outputs = []
        i = 0
        while i < min(args.num_examples, len(dataset)):
            dataset_batch = dataset[i : min(args.num_examples, i + args.batch_size)]
            batch_inputs = []
            for text_input, ground_truth_output in dataset_batch:
                attacked_text = textattack.shared.AttackedText(text_input)
                batch_inputs.append(attacked_text.tokenizer_input)
                ground_truth_outputs.append(ground_truth_output)
            batch_preds = model(batch_inputs)

            if not isinstance(batch_preds, torch.Tensor):
                batch_preds = torch.Tensor(batch_preds)

            preds.extend(batch_preds)
            i += args.batch_size

        preds = torch.stack(preds).squeeze().cpu()
        ground_truth_outputs = torch.tensor(ground_truth_outputs).cpu()

        logger.info(f"Got {len(preds)} predictions.")

        if preds.ndim == 1:
            # if preds is just a list of numbers, assume regression for now
            # TODO integrate with `textattack.metrics` package
            pearson_correlation, _ = scipy.stats.pearsonr(ground_truth_outputs, preds)
            spearman_correlation, _ = scipy.stats.spearmanr(ground_truth_outputs, preds)

            logger.info(f"Pearson correlation = {_cb(pearson_correlation)}")
            logger.info(f"Spearman correlation = {_cb(spearman_correlation)}")
        else:
            guess_labels = preds.argmax(dim=1)
            successes = (guess_labels == ground_truth_outputs).sum().item()
            perc_accuracy = successes / len(preds) * 100.0
            perc_accuracy = "{:.2f}%".format(perc_accuracy)
            logger.info(f"Correct {successes}/{len(preds)} ({_cb(perc_accuracy)})")

            if not args.do_not_push:
                # saving correct [differential] inputs (v2 adv examples that v1 predicts correctly)
                mask = (guess_labels == ground_truth_outputs) # tensor equivalence checking, not list
                correct_indices = mask.nonzero(as_tuple=True)[0].tolist()
                # print(correct_indices)
                correct_dataset = dataset._dataset.select(correct_indices)
                path_to_load = args.dataset_from_huggingface + "_original"
                if "adv" in args.dataset_from_huggingface:
                    # path_to_load = args.dataset_from_huggingface[:-4] + "_original" + "_adv"
                    path_to_load = args.dataset_from_huggingface[:-len("_advtraining")] + "_original" + "_advtraining"
                # the dataset only contains train split
                corresponding_dataset = load_dataset(path_to_load)['train'].select(correct_indices)

                save_path = args.model.replace("best_model", "differential_examples") + "/" + args.dataset_from_huggingface.split("_")[2]
                # print(save_path)
                original_save_path = save_path + "_original"
                if "adv" in args.dataset_from_huggingface:
                    save_path = save_path + "_adv"
                    original_save_path = original_save_path + "_adv"
                if "_var" in args.dataset_from_huggingface:
                    save_path = save_path + "_var"
                    original_save_path = original_save_path + "_var"
                correct_dataset.save_to_disk(save_path)
                corresponding_dataset.save_to_disk(original_save_path)

                try:
                    dataset_path = args.dataset_from_huggingface
                    model_dataset = dataset_path.split('/')[1].split("_") # e.g.: [albertbasev2, imdb, pwws]
                    dataset_name = model_dataset[0][:-2] + "_" + model_dataset[1] + "_" + model_dataset[2] + "_differential"
                    original_dataset_name = dataset_name + "_original"
                    if "adv" in args.dataset_from_huggingface:
                        dataset_name = dataset_name + "_adv"
                        original_dataset_name = original_dataset_name + "_adv"
                    # if "_var" in args.dataset_from_huggingface:
                    #     dataset_name = dataset_name[:-len("_differential")] + "_var" + "_differential"
                    #     original_dataset_name = original_dataset_name[:-len("_differential_original")] + "_var" + "_differential_original"
                    correct_dataset.push_to_hub(dataset_name)
                    corresponding_dataset.push_to_hub(original_dataset_name)
                    print("Successfully pushed the dataset to HuggingFace.")
                except Exception as e:
                    print("Automatic data upload has failed. Please try manual upload using the .ipynb file.")
                    print("Error details:", str(e))

    def run(self, args):
        args = ModelEvalArgs(**vars(args))
        textattack.shared.utils.set_seed(args.random_seed)

        # Default to 'all' if no model chosen.
        if not (args.model or args.model_from_huggingface or args.model_from_file):
            for model_name in list(HUGGINGFACE_MODELS.keys()) + list(
                TEXTATTACK_MODELS.keys()
            ):
                args.model = model_name
                self.test_model_on_dataset(args)
                logger.info("-" * 50)
        else:
            self.test_model_on_dataset(args)

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "eval",
            help="evaluate a model with TextAttack",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )

        parser = ModelArgs._add_parser_args(parser)
        parser = DatasetArgs._add_parser_args(parser)

        parser.add_argument("--random-seed", default=765, type=int)
        parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="The batch size for evaluating the model.",
        )
        parser.add_argument(
            "--num-examples",
            "-n",
            type=int,
            required=False,
            default=5,
            help="The number of examples to process, -1 for entire dataset",
        )
        parser.add_argument(
            "--num-examples-offset",
            "-o",
            type=int,
            required=False,
            default=0,
            help="The offset to start at in the dataset.",
        )

        parser.set_defaults(func=EvalModelCommand())
