"""
Attacker Class
==============
"""

import collections
import hashlib
import json
import logging
import multiprocessing as mp
import os
import queue
import random
import re
import traceback
import time
from datasets import Dataset, Features, ClassLabel, Value

import torch
import tqdm

import textattack
from textattack.attack_results import (
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)
from textattack.shared.utils import logger

from .attack import Attack
from .attack_args import AttackArgs


class Attacker:
    """Class for running attacks on a dataset with specified parameters. This
    class uses the :class:`~textattack.Attack` to actually run the attacks,
    while also providing useful features such as parallel processing,
    saving/resuming from a checkpint, logging to files and stdout.

    Args:
        attack (:class:`~textattack.Attack`):
            :class:`~textattack.Attack` used to actually carry out the attack.
        dataset (:class:`~textattack.datasets.Dataset`):
            Dataset to attack.
        attack_args (:class:`~textattack.AttackArgs`):
            Arguments for attacking the dataset. For default settings, look at the `AttackArgs` class.

    Example::

        >>> import textattack
        >>> import transformers

        >>> model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
        >>> tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
        >>> model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

        >>> attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
        >>> dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")

        >>> # Attack 20 samples with CSV logging and checkpoint saved every 5 interval
        >>> attack_args = textattack.AttackArgs(
        ...     num_examples=20,
        ...     log_to_csv="log.csv",
        ...     checkpoint_interval=5,
        ...     checkpoint_dir="checkpoints",
        ...     disable_stdout=True
        ... )

        >>> attacker = textattack.Attacker(attack, dataset, attack_args)
        >>> attacker.attack_dataset()
    """

    def __init__(self, attack, dataset, attack_args=None):
        assert isinstance(
            attack, Attack
        ), f"`attack` argument must be of type `textattack.Attack`, but got type of `{type(attack)}`."
        assert isinstance(
            dataset, textattack.datasets.Dataset
        ), f"`dataset` must be of type `textattack.datasets.Dataset`, but got type `{type(dataset)}`."

        if attack_args:
            assert isinstance(
                attack_args, AttackArgs
            ), f"`attack_args` must be of type `textattack.AttackArgs`, but got type `{type(attack_args)}`."
        else:
            attack_args = AttackArgs()

        self.attack = attack
        self.dataset = dataset
        self.attack_args = attack_args
        self.attack_log_manager = None
        observer = getattr(self.attack.goal_function, "candidate_observer", None)
        manifest = getattr(self.dataset, "manifest", None)
        if manifest is not None and self.attack.goal_function.model2 is not None:
            self._validate_manifest_model_pair(manifest)
        if observer is not None and manifest is not None:
            # Candidate rows inherit immutable provenance from the exact train
            # manifest used for calibration, not merely CLI display strings.
            observer.metadata.update(
                {
                    "dataset_revision_or_fingerprint": (
                        manifest.dataset_revision_or_fingerprint
                    ),
                    "manifest_seed": manifest.seed,
                    "new_model_id": manifest.new_model_id,
                    "new_model_revision": manifest.new_model_revision,
                    "old_model_id": manifest.old_model_id,
                    "old_model_revision": manifest.old_model_revision,
                }
            )

        # The hash excludes values whose names imply credentials. It identifies
        # behaviorally relevant run settings without risking secret leakage.
        public_config = {
            key: value
            for key, value in vars(attack_args).items()
            if "key" not in key.lower() and "secret" not in key.lower()
        }
        encoded_config = json.dumps(
            public_config, sort_keys=True, default=repr
        ).encode("utf-8")
        self.run_config_hash = hashlib.sha256(encoded_config).hexdigest()

        # This is to be set if loading from a checkpoint
        self._checkpoint = None

    def _validate_manifest_model_pair(self, manifest):
        """Fail before queries if a frozen manifest targets other checkpoints."""

        wrappers = (
            ("new", self.attack.goal_function.model, manifest.new_model_id,
             manifest.new_model_revision,
             getattr(self.attack_args, "model_revision", None)),
            ("old", self.attack.goal_function.model2, manifest.old_model_id,
             manifest.old_model_revision,
             getattr(self.attack_args, "second_model_revision", None)),
        )
        for (
            role,
            wrapper,
            expected_id,
            expected_revision,
            configured_revision,
        ) in wrappers:
            config = getattr(wrapper.model, "config", None)
            if config is None:
                raise ValueError(f"The {role} model has no configuration metadata.")
            actual_id = str(getattr(config, "_name_or_path", ""))
            if os.path.exists(expected_id) or os.path.exists(actual_id):
                expected_comparable = os.path.realpath(expected_id)
                actual_comparable = os.path.realpath(actual_id)
            else:
                expected_comparable = expected_id
                actual_comparable = actual_id
            if actual_comparable and expected_comparable != actual_comparable:
                raise ValueError(
                    f"The {role} model does not match the frozen manifest: "
                    f"{actual_comparable!r} != {expected_comparable!r}."
                )
            actual_revision = (
                getattr(config, "_commit_hash", None)
                or configured_revision
                or actual_id
            )
            # Local checkpoints commonly identify their path as the revision;
            # Hub checkpoints expose an immutable commit hash.
            if expected_revision and actual_revision != expected_revision:
                raise ValueError(
                    f"The {role} model revision does not match the manifest: "
                    f"{actual_revision!r} != {expected_revision!r}."
                )

        label_names = getattr(self.dataset, "label_names", None)
        if label_names:
            config = self.attack.goal_function.model.model.config
            if len(label_names) != int(config.num_labels):
                raise ValueError(
                    "Dataset label count does not match the model label count."
                )
            id2label = getattr(config, "id2label", {}) or {}
            if len(id2label) != int(config.num_labels):
                # A missing semantic mapping still has an unambiguous identity
                # index mapping after the class-count check above.
                return
            model_labels = [
                str(id2label.get(index, id2label.get(str(index)))).strip().lower()
                for index in range(config.num_labels)
            ]
            dataset_labels = [str(label).strip().lower() for label in label_names]
            generic = all(
                re.fullmatch(r"label[_ -]?\d+", label) for label in model_labels
            )
            if not generic and model_labels != dataset_labels:
                raise ValueError(
                    "Dataset label names do not match model id2label order: "
                    f"{dataset_labels!r} != {model_labels!r}."
                )

    def _profiled_constraint(self):
        """Return the single online resource-profiled constraint, if present."""

        profiled = [
            constraint
            for constraint in self.attack.constraints
            if hasattr(constraint, "profile_dict") and hasattr(constraint, "profile")
        ]
        if len(profiled) > 1:
            raise ValueError("Only one profiled constraint is supported per attack.")
        return profiled[0] if profiled else None

    @staticmethod
    def _profile_delta(constraint, before, peak_vram_bytes):
        """Derive per-example NLI counters from cumulative runtime state."""

        after = vars(constraint.profile)
        delta = {
            key: after[key] - before[key]
            for key in after
            if key != "peak_vram_bytes"
        }
        lookups = delta["cache_hits"] + delta["cache_misses"]
        delta["cache_hit_rate"] = (
            delta["cache_hits"] / lookups if lookups else None
        )
        delta["seconds_per_candidate"] = (
            delta["inference_seconds"] / delta["candidates"]
            if delta["candidates"]
            else None
        )
        delta["truncated_directional_pair_rate"] = (
            delta["truncated_directional_pairs"] / delta["directional_pairs"]
            if delta["directional_pairs"]
            else None
        )
        delta["truncated_candidate_rate"] = (
            delta["truncated_candidates"] / delta["candidates"]
            if delta["candidates"]
            else None
        )
        delta["peak_vram_bytes"] = peak_vram_bytes
        return delta

    def _get_worklist(self, start, end, num_examples, shuffle):
        if end - start < num_examples:
            logger.warn(
                f"Attempting to attack {num_examples} samples when only {end-start} are available."
            )
        candidates = list(range(start, end))
        if shuffle:
            random.shuffle(candidates)
        worklist = collections.deque(candidates[:num_examples])
        candidates = collections.deque(candidates[num_examples:])
        assert (len(worklist) + len(candidates)) == (end - start)
        return worklist, candidates

    def _attack(self):
        
        # if self.attack_args.filter_test_by_labels:
        #     print("Entered here for label filtering")
        #     labels_to_keep = self.attack_args.filter_test_by_labels
        #     if not isinstance(labels_to_keep, set):
        #         labels_to_keep = set(labels_to_keep)
        #     self.dataset._dataset = self.dataset._dataset.filter(
        #         lambda x: x[self.dataset.output_column] in labels_to_keep
        #     )
        #     print("Label filtering successful")

        print("Data Format", self.dataset.input_columns)
        """Internal method that carries out attack.

        No parallel processing is involved.
        """
        if torch.cuda.is_available():
            self.attack.cuda_()

        if self._checkpoint:
            num_remaining_attacks = self._checkpoint.num_remaining_attacks
            worklist = self._checkpoint.worklist
            worklist_candidates = self._checkpoint.worklist_candidates
            logger.info(
                f"Recovered from checkpoint previously saved at {self._checkpoint.datetime}."
            )
        else:
            if self.attack_args.num_successful_examples:
                num_remaining_attacks = self.attack_args.num_successful_examples
                # We make `worklist` deque (linked-list) for easy pop and append.
                # Candidates are other samples we can attack if we need more samples.
                worklist, worklist_candidates = self._get_worklist(
                    self.attack_args.num_examples_offset,
                    len(self.dataset),
                    self.attack_args.num_successful_examples,
                    self.attack_args.shuffle,
                )
            else:
                num_remaining_attacks = self.attack_args.num_examples
                # We make `worklist` deque (linked-list) for easy pop and append.
                # Candidates are other samples we can attack if we need more samples.
                worklist, worklist_candidates = self._get_worklist(
                    self.attack_args.num_examples_offset,
                    len(self.dataset),
                    self.attack_args.num_examples,
                    self.attack_args.shuffle,
                )

        if not self.attack_args.silent:
            print(self.attack, "\n")

        pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0, dynamic_ncols=True)
        if self._checkpoint:
            num_results = self._checkpoint.results_count
            num_failures = self._checkpoint.num_failed_attacks
            num_skipped = self._checkpoint.num_skipped_attacks
            num_successes = self._checkpoint.num_successful_attacks
        else:
            num_results = 0
            num_failures = 0
            num_skipped = 0
            num_successes = 0

        sample_exhaustion_warned = False
        
        # to save data for creating baseline algorithms
        texts_imdb = []
        texts_imdb_original = []
        texts_premise = []
        texts_hypothesis = []
        texts_premise_original = []
        texts_hypothesis_original = []
        texts_question1 = []
        texts_question2 = []
        texts_question1_original = []
        texts_question2_original = []
        labels = []

        while worklist:
            idx = worklist.popleft()
            try:
                example, ground_truth_output = self.dataset[idx]
                # print("\n")
                # print("ground_truth:", ground_truth_output)
            except IndexError:
                continue
            example = textattack.shared.AttackedText(example)
            source_index = (
                self.dataset.source_index(idx)
                if hasattr(self.dataset, "source_index")
                else idx
            )
            example.attack_attrs["dataset_index"] = source_index
            example.attack_attrs["run_config_hash"] = self.run_config_hash
            example.attack_attrs["ground_truth_output"] = int(ground_truth_output)
            if self.dataset.label_names is not None:
                example.attack_attrs["label_names"] = self.dataset.label_names
                if int(ground_truth_output) < len(self.dataset.label_names):
                    example.attack_attrs["ground_truth_label_name"] = str(
                        self.dataset.label_names[int(ground_truth_output)]
                    )
            observer = getattr(self.attack.goal_function, "candidate_observer", None)
            if observer is not None and hasattr(observer, "start_example"):
                observer.start_example(
                    dataset_index=source_index,
                    original_text=example,
                    ground_truth_output=ground_truth_output,
                )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            profiled_constraint = self._profiled_constraint()
            profile_before = (
                dict(vars(profiled_constraint.profile))
                if profiled_constraint is not None
                else None
            )
            started_at = time.perf_counter()
            try:
                result = self.attack.attack(example, ground_truth_output)
            except Exception as e:
                raise e
            result.wall_clock_seconds = time.perf_counter() - started_at
            result.peak_vram_bytes = (
                torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else 0
            )
            result.nli_profile = (
                self._profile_delta(
                    profiled_constraint, profile_before, result.peak_vram_bytes
                )
                if profiled_constraint is not None
                else None
            )
            if (
                isinstance(result, SkippedAttackResult) and self.attack_args.attack_n
            ) or (
                not isinstance(result, SuccessfulAttackResult)
                and self.attack_args.num_successful_examples
            ):
                if worklist_candidates:
                    next_sample = worklist_candidates.popleft()
                    worklist.append(next_sample)
                else:
                    if not sample_exhaustion_warned:
                        logger.warn("Ran out of samples to attack!")
                        sample_exhaustion_warned = True
            else:
                pbar.update(1)

            self.attack_log_manager.log_result(result)
            
            # this is extracting the original input: t1 = result.original_result.attacked_text, t1.printable_text()
            # store the successful attack results to form baseline algorithm
            if isinstance(result, (SuccessfulAttackResult, MaximizedAttackResult)):
                txt = result.perturbed_result.attacked_text.printable_text()
                txt_original = result.original_result.attacked_text.printable_text()
                lines = txt.split("\n")
                lines_original = txt_original.split("\n")
                if len(lines) >= 2:
                    if lines[0].startswith("Premise: ") and lines[1].startswith("Hypothesis: "):
                        # if dataset is snli: split premise/hypothesis
                        texts_premise.append(lines[0][len("Premise: "):].strip())
                        texts_hypothesis.append(lines[1][len("Hypothesis: "):].strip())
                        texts_premise_original.append(lines_original[0][len("Premise: "):].strip())
                        texts_hypothesis_original.append(lines_original[1][len("Hypothesis: "):].strip())
                    elif lines[0].startswith("Question1: ") and lines[1].startswith("Question2: "):
                        # if dataset is qqp: split question1/question2
                        texts_question1.append(lines[0][len("Question1: "):].strip())
                        texts_question2.append(lines[1][len("Question2: "):].strip())
                        texts_question1_original.append(lines_original[0][len("Question1: "):].strip())
                        texts_question2_original.append(lines_original[1][len("Question2: "):].strip())
                    else:
                        raise Exception("Sorry, no matching data schema for the current task!")
                    labels.append(ground_truth_output)
                else:
                    # in some cases the splitting is messed up, manually check the result
                    if "qqp" in self.attack_args.dataset_from_huggingface or "snli" in self.attack_args.dataset_from_huggingface or "rte" in self.attack_args.dataset_from_huggingface or "mrpc" in self.attack_args.dataset_from_huggingface:
                        # do not record the data to prevent causing confusion
                        print("Please manually check this case!")
                        print("txt: ", txt)
                        print("txt_original: ", txt_original)
                        print("label: ", ground_truth_output)
                    else:
                        #dataset is imdb/sst2: directly store txt & label (sentence/text, label)
                        texts_imdb.append(txt) # refer to the attacked_text.py: if single sentence: no prefix; multiple: capitalize prefix
                        texts_imdb_original.append(txt_original)
                        labels.append(ground_truth_output)
            # print("all labels: ", labels)

            if not self.attack_args.disable_stdout and not self.attack_args.silent:
                print("\n")
            num_results += 1

            if isinstance(result, SkippedAttackResult):
                num_skipped += 1
            if isinstance(result, (SuccessfulAttackResult, MaximizedAttackResult)):
                num_successes += 1
            if isinstance(result, FailedAttackResult):
                num_failures += 1
            pbar.set_description(
                f"[Succeeded / Failed / Skipped / Total] {num_successes} / {num_failures} / {num_skipped} / {num_results}"
            )

            if (
                self.attack_args.checkpoint_interval
                and len(self.attack_log_manager.results)
                % self.attack_args.checkpoint_interval
                == 0
            ):
                new_checkpoint = textattack.shared.AttackCheckpoint(
                    self.attack_args,
                    self.attack_log_manager,
                    worklist,
                    worklist_candidates,
                )
                new_checkpoint.save()
                self.attack_log_manager.flush()

        # create the dataset of successful attacks
        if texts_imdb:
            features = Features({'text': Value('string'),
                                 'label': Value('int32')})
            dataset = Dataset.from_dict({
                'text': texts_imdb,
                'label': labels
            }, features=features)
            dataset_original = Dataset.from_dict({
                'text': texts_imdb_original,
                'label': labels
            }, features=features)
        elif texts_hypothesis:
            features = Features({'premise': Value('string'),
                                 'hypothesis': Value('string'),
                                 'label': Value('int32')})
            dataset = Dataset.from_dict({
                'premise': texts_premise,
                'hypothesis': texts_hypothesis,
                'label': labels
            }, features=features)
            dataset_original = Dataset.from_dict({
                'premise': texts_premise_original,
                'hypothesis': texts_hypothesis_original,
                'label': labels
            }, features=features)
        elif texts_question1:
            features = Features({'question1': Value('string'),
                                 'question2': Value('string'),
                                 'label': Value('int32')})
            dataset = Dataset.from_dict({
                'question1': texts_question1,
                'question2': texts_question2,
                'label': labels
            }, features=features)
            dataset_original = Dataset.from_dict({
                'question1': texts_question1_original,
                'question2': texts_question2_original,
                'label': labels
            }, features=features)
        
        if not self.attack_args.do_not_push:
            # save the dataset with info from self.attack_args
            atk_method = self.attack_args.attack_recipe
            model_path = self.attack_args.model
            save_path = model_path.replace("best_model", "successful_examples") + "/" + atk_method
            if self.attack_args.attack_recipe == "pair":
                save_path = save_path + "_" + self.attack_args.base_recipe
                if self.attack_args.lambda1 and self.attack_args.lambda2:
                    save_path = save_path + "_" + str(self.attack_args.lambda1) + "_" + str(self.attack_args.lambda2)
            save_path_original = save_path + "_original"
            if "adv" in self.attack_args.dataset_from_huggingface:
                partition = self.attack_args.dataset_from_huggingface.split("_")[-1]
                save_path = save_path + "_adv" + partition
                save_path_original = save_path_original + "_adv" + partition
            dataset.save_to_disk(save_path)
            dataset_original.save_to_disk(save_path_original)
            # ../outputs/albertbasev2_imdb/best_model --> albertbasev2_imdb_pwws
            push_dir = model_path.split("/")[2] + "_" + atk_method
            if self.attack_args.attack_recipe == "pair":
                push_dir = push_dir + "_" + self.attack_args.base_recipe
                if self.attack_args.lambda1 and self.attack_args.lambda2:
                    push_dir = push_dir + "_" + str(self.attack_args.lambda1) + "_" + str(self.attack_args.lambda2)
            push_dir_original = push_dir + "_original"
            if "adv" in self.attack_args.dataset_from_huggingface:
                print("this dataset is for adversarial training")
                partition = self.attack_args.dataset_from_huggingface.split("_")[-1]
                push_dir = push_dir + "_adv" + partition
                push_dir_original = push_dir_original + "_adv" + partition
            try:
                dataset.push_to_hub(push_dir) # defaults to personal account
                dataset_original.push_to_hub(push_dir_original)
            except Exception as e:
                print("Automatic data upload has failed. Please try manual upload using the .ipynb file.")
                print("Error details:", str(e))

        pbar.close()
        print()
        # Enable summary stdout
        if not self.attack_args.silent and self.attack_args.disable_stdout:
            self.attack_log_manager.enable_stdout()

        if self.attack_args.enable_advance_metrics:
            self.attack_log_manager.enable_advance_metrics = True

        self.attack_log_manager.log_summary()
        self.attack_log_manager.flush()
        print()

    def _attack_parallel(self):
        pytorch_multiprocessing_workaround()

        if self._checkpoint:
            num_remaining_attacks = self._checkpoint.num_remaining_attacks
            worklist = self._checkpoint.worklist
            worklist_candidates = self._checkpoint.worklist_candidates
            logger.info(
                f"Recovered from checkpoint previously saved at {self._checkpoint.datetime}."
            )
        else:
            if self.attack_args.num_successful_examples:
                num_remaining_attacks = self.attack_args.num_successful_examples
                # We make `worklist` deque (linked-list) for easy pop and append.
                # Candidates are other samples we can attack if we need more samples.
                worklist, worklist_candidates = self._get_worklist(
                    self.attack_args.num_examples_offset,
                    len(self.dataset),
                    self.attack_args.num_successful_examples,
                    self.attack_args.shuffle,
                )
            else:
                num_remaining_attacks = self.attack_args.num_examples
                # We make `worklist` deque (linked-list) for easy pop and append.
                # Candidates are other samples we can attack if we need more samples.
                worklist, worklist_candidates = self._get_worklist(
                    self.attack_args.num_examples_offset,
                    len(self.dataset),
                    self.attack_args.num_examples,
                    self.attack_args.shuffle,
                )

        in_queue = torch.multiprocessing.Queue()
        out_queue = torch.multiprocessing.Queue()
        for i in worklist:
            try:
                example, ground_truth_output = self.dataset[i]
                example = textattack.shared.AttackedText(example)
                if self.dataset.label_names is not None:
                    example.attack_attrs["label_names"] = self.dataset.label_names
                in_queue.put((i, example, ground_truth_output))
            except IndexError:
                raise IndexError(
                    f"Tried to access element at {i} in dataset of size {len(self.dataset)}."
                )

        # We reserve the first GPU for coordinating workers.
        num_gpus = torch.cuda.device_count()
        num_workers = self.attack_args.num_workers_per_device * num_gpus
        logger.info(f"Running {num_workers} worker(s) on {num_gpus} GPU(s).")

        # Lock for synchronization
        lock = mp.Lock()

        # We move Attacker (and its components) to CPU b/c we don't want models using wrong GPU in worker processes.
        self.attack.cpu_()
        torch.cuda.empty_cache()

        # Start workers.
        worker_pool = torch.multiprocessing.Pool(
            num_workers,
            attack_from_queue,
            (
                self.attack,
                self.attack_args,
                num_gpus,
                mp.Value("i", 1, lock=False),
                lock,
                in_queue,
                out_queue,
            ),
        )

        # Log results asynchronously and update progress bar.
        if self._checkpoint:
            num_results = self._checkpoint.results_count
            num_failures = self._checkpoint.num_failed_attacks
            num_skipped = self._checkpoint.num_skipped_attacks
            num_successes = self._checkpoint.num_successful_attacks
        else:
            num_results = 0
            num_failures = 0
            num_skipped = 0
            num_successes = 0

        logger.info(f"Worklist size: {len(worklist)}")
        logger.info(f"Worklist candidate size: {len(worklist_candidates)}")

        sample_exhaustion_warned = False
        pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0, dynamic_ncols=True)

        # to save data for creating baseline algorithms
        texts_imdb = []
        texts_imdb_original = []
        texts_premise = []
        texts_hypothesis = []
        texts_premise_original = []
        texts_hypothesis_original = []
        texts_question1 = []
        texts_question2 = []
        texts_question1_original = []
        texts_question2_original = []
        labels = []

        while worklist:
            # modified accordingly the attack_from_queue function
            idx, result, current_label = out_queue.get(block=True)
            worklist.remove(idx)

            if isinstance(result, tuple) and isinstance(result[0], Exception):
                logger.error(
                    f'Exception encountered for input "{self.dataset[idx][0]}".'
                )
                error_trace = result[1]
                logger.error(error_trace)
                in_queue.close()
                in_queue.join_thread()
                out_queue.close()
                out_queue.join_thread()
                worker_pool.terminate()
                worker_pool.join()
                return
            elif (
                isinstance(result, SkippedAttackResult) and self.attack_args.attack_n
            ) or (
                not isinstance(result, SuccessfulAttackResult)
                and self.attack_args.num_successful_examples
            ): # this means in situations such as you want n successful examples, then you're gonna attack additional examples if one fails/is skipped
                if worklist_candidates:
                    next_sample = worklist_candidates.popleft()
                    example, ground_truth_output = self.dataset[next_sample]
                    example = textattack.shared.AttackedText(example)
                    if self.dataset.label_names is not None:
                        example.attack_attrs["label_names"] = self.dataset.label_names
                    worklist.append(next_sample) # expanding the worklist
                    in_queue.put((next_sample, example, ground_truth_output))
                else:
                    if not sample_exhaustion_warned:
                        logger.warn("Ran out of samples to attack!")
                        sample_exhaustion_warned = True
            else:
                pbar.update()

            try:
                self.attack_log_manager.log_result(result)
            except:
                print("error occurred during result logging, please manually check the result")

            # this is extracting the original input: t1 = result.original_result.attacked_text, t1.printable_text()
            # store the successful attack results to form baseline algorithm
            if isinstance(result, (SuccessfulAttackResult, MaximizedAttackResult)):
                txt = result.perturbed_result.attacked_text.printable_text()
                txt_original = result.original_result.attacked_text.printable_text()
                lines = txt.split("\n")
                lines_original = txt_original.split("\n")
                if len(lines) >= 2:
                    if lines[0].startswith("Premise: ") and lines[1].startswith("Hypothesis: "):
                        # if dataset is snli: split premise/hypothesis
                        texts_premise.append(lines[0][len("Premise: "):].strip())
                        texts_hypothesis.append(lines[1][len("Hypothesis: "):].strip())
                        texts_premise_original.append(lines_original[0][len("Premise: "):].strip())
                        texts_hypothesis_original.append(lines_original[1][len("Hypothesis: "):].strip())
                    elif lines[0].startswith("Question1: ") and lines[1].startswith("Question2: "):
                        # if dataset is qqp: split question1/question2
                        texts_question1.append(lines[0][len("Question1: "):].strip())
                        texts_question2.append(lines[1][len("Question2: "):].strip())
                        texts_question1_original.append(lines_original[0][len("Question1: "):].strip())
                        texts_question2_original.append(lines_original[1][len("Question2: "):].strip())
                    else:
                        raise Exception("Sorry, no matching data schema for the current task!")
                    labels.append(current_label)
                else:
                    # in some cases the splitting is messed up, manually check the result
                    if "qqp" in self.attack_args.dataset_from_huggingface or "snli" in self.attack_args.dataset_from_huggingface or "rte" in self.attack_args.dataset_from_huggingface or "mrpc" in self.attack_args.dataset_from_huggingface:
                        # do not record the data to prevent causing confusion
                        print("Please manually check this case!")
                        print("txt: ", txt)
                        print("txt_original: ", txt_original)
                        print("label: ", current_label)
                    else:
                        #dataset is imdb/sst2: directly store txt & label (sentence/text, label)
                        texts_imdb.append(txt) # refer to the attacked_text.py: if single sentence: no prefix; multiple: capitalize prefix
                        texts_imdb_original.append(txt_original)
                        labels.append(current_label)
                # should not append ground_truth_output: it's the label of the last sample to attack (stored during queuing) or the label of additional samples (due to the use of attack_n/num_successful arguments)
                # labels.append(current_label)
            # print("all labels: ", labels)

            num_results += 1

            if isinstance(result, SkippedAttackResult):
                num_skipped += 1
            if isinstance(result, (SuccessfulAttackResult, MaximizedAttackResult)):
                num_successes += 1
            if isinstance(result, FailedAttackResult):
                num_failures += 1
            pbar.set_description(
                f"[Succeeded / Failed / Skipped / Total] {num_successes} / {num_failures} / {num_skipped} / {num_results}"
            )

            if (
                self.attack_args.checkpoint_interval
                and len(self.attack_log_manager.results)
                % self.attack_args.checkpoint_interval
                == 0
            ):
                new_checkpoint = textattack.shared.AttackCheckpoint(
                    self.attack_args,
                    self.attack_log_manager,
                    worklist,
                    worklist_candidates,
                )
                new_checkpoint.save()
                self.attack_log_manager.flush()

        # Send sentinel values to worker processes
        for _ in range(num_workers):
            in_queue.put(("END", "END", "END"))
        worker_pool.close()
        worker_pool.join()

        # create the dataset of successful attacks
        if texts_imdb:
            features = Features({'text': Value('string'),
                                 'label': Value('int32')})
            dataset = Dataset.from_dict({
                'text': texts_imdb,
                'label': labels
            }, features=features)
            dataset_original = Dataset.from_dict({
                'text': texts_imdb_original,
                'label': labels
            }, features=features)
        elif texts_hypothesis:
            features = Features({'premise': Value('string'),
                                 'hypothesis': Value('string'),
                                 'label': Value('int32')})
            dataset = Dataset.from_dict({
                'premise': texts_premise,
                'hypothesis': texts_hypothesis,
                'label': labels
            }, features=features)
            dataset_original = Dataset.from_dict({
                'premise': texts_premise_original,
                'hypothesis': texts_hypothesis_original,
                'label': labels
            }, features=features)
        elif texts_question1:
            features = Features({'question1': Value('string'),
                                 'question2': Value('string'),
                                 'label': Value('int32')})
            dataset = Dataset.from_dict({
                'question1': texts_question1,
                'question2': texts_question2,
                'label': labels
            }, features=features)
            dataset_original = Dataset.from_dict({
                'question1': texts_question1_original,
                'question2': texts_question2_original,
                'label': labels
            }, features=features)
        
        if not self.attack_args.do_not_push:
            # save the dataset with info from self.attack_args
            atk_method = self.attack_args.attack_recipe
            model_path = self.attack_args.model
            save_path = model_path.replace("best_model", "successful_examples") + "/" + atk_method
            if self.attack_args.attack_recipe == "pair":
                save_path = save_path + "_" + self.attack_args.base_recipe
                if self.attack_args.lambda1 and self.attack_args.lambda2:
                    save_path = save_path + "_" + str(self.attack_args.lambda1) + "_" + str(self.attack_args.lambda2)
            save_path_original = save_path + "_original"
            if "adv" in self.attack_args.dataset_from_huggingface:
                partition = self.attack_args.dataset_from_huggingface.split("_")[-1]
                save_path = save_path + "_adv" + partition
                save_path_original = save_path_original + "_adv" + partition
            dataset.save_to_disk(save_path)
            dataset_original.save_to_disk(save_path_original)
            # ../outputs/albertbasev2_imdb/best_model --> albertbasev2_imdb_pwws
            push_dir = model_path.split("/")[2] + "_" + atk_method
            if self.attack_args.attack_recipe == "pair":
                push_dir = push_dir + "_" + self.attack_args.base_recipe
                if self.attack_args.lambda1 and self.attack_args.lambda2:
                    push_dir = push_dir + "_" + str(self.attack_args.lambda1) + "_" + str(self.attack_args.lambda2)
            push_dir_original = push_dir + "_original"
            if "adv" in self.attack_args.dataset_from_huggingface:
                print("this dataset is for adversarial training")
                partition = self.attack_args.dataset_from_huggingface.split("_")[-1]
                push_dir = push_dir + "_adv" + partition
                push_dir_original = push_dir_original + "_adv" + partition
            try:
                dataset.push_to_hub(push_dir) # defaults to personal account
                dataset_original.push_to_hub(push_dir_original)
            except Exception as e:
                print("Automatic data upload has failed. Please try manual upload using the .ipynb file.")
                print("Error details:", str(e))

        pbar.close()
        print()
        # Enable summary stdout.
        if not self.attack_args.silent and self.attack_args.disable_stdout:
            self.attack_log_manager.enable_stdout()

        if self.attack_args.enable_advance_metrics:
            self.attack_log_manager.enable_advance_metrics = True

        self.attack_log_manager.log_summary()
        self.attack_log_manager.flush()
        print()

    def attack_dataset(self):
        """Attack the dataset.

        Returns:
            :obj:`list[AttackResult]` - List of :class:`~textattack.attack_results.AttackResult` obtained after attacking the given dataset..
        """
        if self.attack_args.silent:
            logger.setLevel(logging.ERROR)

        if self.attack_args.query_budget:
            self.attack.goal_function.query_budget = self.attack_args.query_budget

        if self.attack_args.parallel and getattr(
            self.attack.goal_function, "candidate_observer", None
        ):
            # Candidate collection is intentionally serial so one append-only
            # stream has deterministic order and no cross-process corruption.
            raise ValueError("Candidate observation cannot run with --parallel.")
        if self.attack_args.parallel and self._profiled_constraint() is not None:
            # Worker-local NLI counters cannot be merged by the legacy
            # multiprocessing path without losing cache and timing semantics.
            raise ValueError("Profiled SemDT runs currently require serial execution.")

        if not self.attack_log_manager:
            self.attack_log_manager = AttackArgs.create_loggers_from_args(
                self.attack_args
            )

        textattack.shared.utils.set_seed(self.attack_args.random_seed)
        if self.dataset.shuffled and self.attack_args.checkpoint_interval:
            # Not allowed b/c we cannot recover order of shuffled data
            raise ValueError(
                "Cannot use `--checkpoint-interval` with dataset that has been internally shuffled."
            )

        self.attack_args.num_examples = (
            len(self.dataset)
            if self.attack_args.num_examples == -1
            else self.attack_args.num_examples
        )
        if self.attack_args.parallel:
            if torch.cuda.device_count() == 0:
                raise Exception(
                    "Found no GPU on your system. To run attacks in parallel, GPU is required."
                )
            self._attack_parallel()
        else:
            self._attack()

        self._write_nli_profile()
        if self.attack_args.silent:
            logger.setLevel(logging.INFO)

        return self.attack_log_manager.results

    def _write_nli_profile(self):
        """Persist online NLI costs after all serial attack examples finish."""

        output_path = getattr(self.attack_args, "nli_profile_output", None)
        if not output_path:
            return
        constraint = self._profiled_constraint()
        if constraint is None:
            raise ValueError(
                "An NLI profile output requires exactly one profiled constraint."
            )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(
                constraint.profile_dict(),
                handle,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )

    def update_attack_args(self, **kwargs):
        """To update any attack args, pass the new argument as keyword argument
        to this function.

        Examples::

        >>> attacker = #some instance of Attacker
        >>> # To switch to parallel mode and increase checkpoint interval from 100 to 500
        >>> attacker.update_attack_args(parallel=True, checkpoint_interval=500)
        """
        for k in kwargs:
            if hasattr(self.attack_args, k):
                self.attack_args.k = kwargs[k]
            else:
                raise ValueError(f"`textattack.AttackArgs` does not have field {k}.")

    @classmethod
    def from_checkpoint(cls, attack, dataset, checkpoint):
        """Resume attacking from a saved checkpoint. Attacker and dataset must
        be recovered by the user again, while attack args are loaded from the
        saved checkpoint.

        Args:
            attack (:class:`~textattack.Attack`):
                Attack object for carrying out the attack.
            dataset (:class:`~textattack.datasets.Dataset`):
                Dataset to attack.
            checkpoint (:obj:`Union[str, :class:`~textattack.shared.AttackChecpoint`]`):
                Path of saved checkpoint or the actual saved checkpoint.
        """
        assert isinstance(
            checkpoint, (str, textattack.shared.AttackCheckpoint)
        ), f"`checkpoint` must be of type `str` or `textattack.shared.AttackCheckpoint`, but got type `{type(checkpoint)}`."

        if isinstance(checkpoint, str):
            checkpoint = textattack.shared.AttackCheckpoint.load(checkpoint)
        attacker = cls(attack, dataset, checkpoint.attack_args)
        attacker.attack_log_manager = checkpoint.attack_log_manager
        attacker._checkpoint = checkpoint
        return attacker

    @staticmethod
    def attack_interactive(attack):
        print(attack, "\n")

        print("Running in interactive mode")
        print("----------------------------")

        while True:
            print('Enter a sentence to attack or "q" to quit:')
            text = input()

            if text == "q":
                break

            if not text:
                continue

            print("Attacking...")

            example = textattack.shared.attacked_text.AttackedText(text)
            output = attack.goal_function.get_output(example)
            result = attack.attack(example, output)
            print(result.__str__(color_method="ansi") + "\n")


#
# Helper Methods for multiprocess attacks
#
def pytorch_multiprocessing_workaround():
    # This is a fix for a known bug
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass


def set_env_variables(gpu_id):
    # Disable tensorflow logs, except in the case of an error.
    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Set sharing strategy to file_system to avoid file descriptor leaks
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Only use one GPU, if we have one.
    # For Tensorflow
    # TODO: Using USE with `--parallel` raises similar issue as https://github.com/tensorflow/tensorflow/issues/38518#
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # For PyTorch
    torch.cuda.set_device(gpu_id)

    # Fix TensorFlow GPU memory growth
    try:
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                gpu = gpus[gpu_id]
                tf.config.experimental.set_visible_devices(gpu, "GPU")
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    except ModuleNotFoundError:
        pass


def attack_from_queue(
    attack, attack_args, num_gpus, first_to_start, lock, in_queue, out_queue
):
    assert isinstance(
        attack, Attack
    ), f"`attack` must be of type `Attack`, but got type `{type(attack)}`."

    gpu_id = (torch.multiprocessing.current_process()._identity[0] - 1) % num_gpus
    set_env_variables(gpu_id)
    textattack.shared.utils.set_seed(attack_args.random_seed)
    if torch.multiprocessing.current_process()._identity[0] > 1:
        logging.disable()

    attack.cuda_()

    # Simple non-synchronized check to see if it's the first process to reach this point.
    # This let us avoid waiting for lock.
    if bool(first_to_start.value):
        # If it's first process to reach this step, we first try to acquire the lock to update the value.
        with lock:
            # Because another process could have changed `first_to_start=False` while we wait, we check again.
            if bool(first_to_start.value):
                first_to_start.value = 0
                if not attack_args.silent:
                    print(attack, "\n")

    while True:
        try:
            i, example, ground_truth_output = in_queue.get(timeout=5)
            if i == "END" and example == "END" and ground_truth_output == "END":
                # End process when sentinel value is received
                break
            else:
                # additionally saving ground_truth_label
                result = attack.attack(example, ground_truth_output)
                out_queue.put((i, result, ground_truth_output))
        except Exception as e:
            if isinstance(e, queue.Empty):
                continue
            else:
                out_queue.put((i, (e, traceback.format_exc()), ground_truth_output))
