"""Resumable command-line stages for SemDT threshold calibration."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict
import hashlib
import json
import os

from textattack.commands import TextAttackCommand
from textattack.semantic_validation.candidate_collection import (
    freeze_calibration_split,
    score_candidate_records,
    stratified_candidate_sample,
    supplemental_audit_sample,
    write_split_manifest,
)
from textattack.semantic_validation.distribution_audit import (
    audit_fixed_threshold,
    distribution_shift_report,
    judge_agreement,
    sample_trajectory_audit,
)
from textattack.semantic_validation.judges import load_judge_config
from textattack.semantic_validation.schemas import (
    CandidateRecord,
    JudgeExample,
    read_jsonl,
    write_jsonl,
)
from textattack.semantic_validation.threshold_search import (
    ThresholdArtifact,
    WeightedSemanticExample,
    needs_supplemental_audit,
    search_thresholds,
    validation_report,
)


# Search implementations are selected by the configured method name.
THRESHOLD_SEARCHERS = {"grid": search_thresholds}


def _ensure_parent(path):
    """Create an output parent without imposing a global run layout."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _write_json(path, payload):
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)


def _require(args, *names):
    """Fail with a stage-specific message instead of an argparse mystery."""

    missing = [f"--{name.replace('_', '-')}" for name in names if not getattr(args, name)]
    if missing:
        raise ValueError(
            f"Stage {args.stage!r} requires: {', '.join(missing)}."
        )


def _candidate_records(path):
    return [CandidateRecord.from_dict(row) for row in read_jsonl(path)]


def _judge_for_config(config):
    """Instantiate exactly one selected backend for this annotation run."""

    if config.backend == "openai":
        from textattack.semantic_validation.judges import OpenAIResponsesJudge

        return OpenAIResponsesJudge(config)
    from textattack.semantic_validation.judges import HuggingFaceCausalLMJudge

    return HuggingFaceCausalLMJudge(config)


def _judge_example(row, task_definition, dataset_override=None):
    """Accept both Base CandidateRecord rows and SemDT trajectory rows."""

    label = row.get("ground_truth_label")
    if label is None:
        raise ValueError(
            f"Candidate {row.get('candidate_id')} has no ground-truth label."
        )
    return JudgeExample(
        candidate_id=row["candidate_id"],
        dataset=dataset_override or str(row.get("dataset", "")),
        task_definition=task_definition,
        ground_truth_label=int(label),
        label_name=str(row.get("label_name", label)),
        original_fields=dict(row["original_fields"]),
        candidate_fields=dict(row["candidate_fields"]),
    )


def _successful_labels(path):
    """Return successful labels and keep terminal failures out of tuning."""

    rows = read_jsonl(path)
    labels = {
        row["candidate_id"]: bool(row["semantic_preserved"])
        for row in rows
        if row.get("success") and row.get("semantic_preserved") is not None
    }
    return labels, rows


def _weighted_examples(candidates, labels):
    return [
        WeightedSemanticExample(
            candidate_id=candidate.candidate_id,
            entailment_score=float(candidate.entailment_score),
            contradiction_score=float(candidate.contradiction_score),
            semantic_preserved=labels[candidate.candidate_id],
            weight=float(candidate.inclusion_weight or 1.0),
        )
        for candidate in candidates
        if candidate.candidate_id in labels
    ]


class SemDTCalibrationCommand(TextAttackCommand):
    """Run one explicit, restartable calibration or audit stage."""

    def run(self, args):
        handlers = {
            "nli-score": self._nli_score,
            "freeze-split": self._freeze_split,
            "annotate": self._annotate,
            "tune-validate": self._tune_validate,
            "supplemental-audit": self._supplemental_audit,
            "frozen-report": self._frozen_report,
            "trajectory-sample": self._trajectory_sample,
            "trajectory-report": self._trajectory_report,
            "judge-agreement": self._judge_agreement,
        }
        handlers[args.stage](args)

    def _nli_score(self, args):
        _require(args, "input", "output")
        from textattack.constraints.semantics import BidirectionalNLI

        candidates = _candidate_records(args.input)
        scorer = BidirectionalNLI(
            args.nli_model,
            model_revision=args.nli_model_revision,
            tokenizer_revision=args.nli_tokenizer_revision,
            device=args.nli_device,
            dtype=args.nli_dtype,
            batch_size=args.nli_batch_size,
            max_length=args.nli_max_length,
            truncation_strategy=args.nli_truncation_strategy,
        )
        scored = score_candidate_records(
            candidates,
            scorer,
            candidate_batch_size=args.candidate_batch_size,
        )
        _ensure_parent(args.output)
        write_jsonl(args.output, (row.to_dict() for row in scored))
        _write_json(
            args.profile_output or f"{args.output}.profile.json",
            scorer.profile_dict(),
        )

    def _freeze_split(self, args):
        _require(args, "input", "output_dir")
        candidates = _candidate_records(args.input)
        sampled = stratified_candidate_sample(
            candidates, sample_size=args.sample_size, seed=args.seed
        )
        search, validation = freeze_calibration_split(
            sampled, search_size=args.search_size, seed=args.seed
        )
        os.makedirs(args.output_dir, exist_ok=True)
        paths = {
            "sampled": os.path.join(args.output_dir, "sampled.jsonl"),
            "search": os.path.join(args.output_dir, "search.jsonl"),
            "validation": os.path.join(args.output_dir, "validation.jsonl"),
            "manifest": os.path.join(args.output_dir, "split_manifest.json"),
        }
        write_jsonl(paths["sampled"], (row.to_dict() for row in sampled))
        write_jsonl(paths["search"], (row.to_dict() for row in search))
        write_jsonl(paths["validation"], (row.to_dict() for row in validation))
        write_split_manifest(
            paths["manifest"], search, validation, seed=args.seed
        )

    def _annotate(self, args):
        _require(args, "input", "output", "judge_config", "task_definition")
        config = load_judge_config(args.judge_config)
        judge = _judge_for_config(config)
        rows = read_jsonl(args.input)
        existing_rows = read_jsonl(args.output) if os.path.exists(args.output) else []
        if any(row.get("backend") != config.backend for row in existing_rows):
            raise ValueError("Existing annotations use a different judge backend.")
        if any(row.get("model") != config.model for row in existing_rows):
            raise ValueError("Existing annotations use a different judge model.")
        seen = {row["candidate_id"] for row in existing_rows}
        pending = [
            _judge_example(row, args.task_definition, args.dataset)
            for row in rows
            if row["candidate_id"] not in seen
        ]
        if args.limit is not None:
            pending = pending[: args.limit]
        _ensure_parent(args.output)
        # Append each outer batch immediately so a long API/HF run is resumable.
        batch_size = max(1, args.annotation_batch_size)
        with open(args.output, "a", encoding="utf-8") as handle:
            for start in range(0, len(pending), batch_size):
                results = judge.annotate(pending[start : start + batch_size])
                for result in results:
                    handle.write(
                        json.dumps(
                            result.to_dict(), ensure_ascii=True, sort_keys=True
                        )
                        + "\n"
                    )
                handle.flush()

    def _tune_validate(self, args):
        _require(
            args,
            "search_candidates",
            "validation_candidates",
            "search_labels",
            "validation_labels",
            "split_manifest",
            "threshold_output",
            "report_output",
            "threshold_search_method",
        )
        search_candidates = _candidate_records(args.search_candidates)
        validation_candidates = _candidate_records(args.validation_candidates)
        with open(args.split_manifest, encoding="utf-8") as handle:
            split_manifest = json.load(handle)
        if split_manifest["search_ids"] != [
            candidate.candidate_id for candidate in search_candidates
        ]:
            raise ValueError("Search candidates do not match the frozen manifest.")
        if split_manifest["validation_ids"] != [
            candidate.candidate_id for candidate in validation_candidates
        ]:
            raise ValueError("Validation candidates do not match the frozen manifest.")
        nli_configs = {
            json.dumps(
                candidate.metadata.get("nli_config"),
                ensure_ascii=True,
                sort_keys=True,
            )
            for candidate in search_candidates + validation_candidates
        }
        if len(nli_configs) != 1 or nli_configs == {"null"}:
            raise ValueError(
                "Search and validation candidates must share one NLI configuration."
            )
        search_labels, search_rows = _successful_labels(args.search_labels)
        validation_labels, validation_rows = _successful_labels(
            args.validation_labels
        )
        candidate_ids = {
            candidate.candidate_id
            for candidate in search_candidates + validation_candidates
        }
        annotation_ids = {
            row["candidate_id"] for row in search_rows + validation_rows
        }
        if not annotation_ids <= candidate_ids:
            raise ValueError("Annotation files contain IDs outside the frozen split.")
        judge_pairs = {
            (row.get("backend"), row.get("model"))
            for row in search_rows + validation_rows
        }
        if len(judge_pairs) != 1:
            raise ValueError("Threshold tuning cannot mix judge backends or models.")
        search_examples = _weighted_examples(search_candidates, search_labels)
        validation_examples = _weighted_examples(
            validation_candidates, validation_labels
        )
        if not validation_examples:
            raise ValueError("No successful validation annotations are available.")
        searcher = THRESHOLD_SEARCHERS[args.threshold_search_method]
        entailment, contradiction, search_metrics = searcher(
            search_examples,
            min_precision=args.min_precision,
            step=args.threshold_step,
        )
        with open(args.split_manifest, "rb") as handle:
            split_hash = hashlib.sha256(handle.read()).hexdigest()
        all_label_rows = search_rows + validation_rows
        first = next(
            (row for row in all_label_rows if row.get("backend") and row.get("model")),
            None,
        )
        if first is None:
            raise ValueError("Judge metadata is absent from annotation files.")
        artifact = ThresholdArtifact(
            entailment_threshold=entailment,
            contradiction_threshold=contradiction,
            min_precision=args.min_precision,
            search_metrics=search_metrics,
            judge_backend=first["backend"],
            judge_model=first["model"],
            dataset=args.dataset or "",
            split_manifest_hash=split_hash,
            nli_config=(
                search_candidates[0].metadata.get("nli_config")
                if search_candidates
                else None
            ),
            threshold_search_method=args.threshold_search_method,
            threshold_step=args.threshold_step,
        )
        _ensure_parent(args.threshold_output)
        artifact.save(args.threshold_output)
        report = validation_report(
            validation_examples,
            entailment,
            contradiction,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        report.update(
            {
                "search_annotation_count": len(search_examples),
                "search_annotation_failures": len(search_rows)
                - len(search_examples),
                "validation_annotation_count": len(validation_examples),
                "validation_annotation_failures": len(validation_rows)
                - len(validation_examples),
                "needs_supplemental_audit": needs_supplemental_audit(
                    validation_examples,
                    minimum_positive=args.minimum_validation_positives,
                ),
                "threshold_search": {
                    "method": args.threshold_search_method,
                    "step": args.threshold_step,
                    "min_precision": args.min_precision,
                },
                "threshold_file": args.threshold_output,
            }
        )
        _write_json(args.report_output, report)

    def _supplemental_audit(self, args):
        _require(
            args,
            "input",
            "sampled_candidates",
            "validation_labels",
            "output",
        )
        validation_labels, _ = _successful_labels(args.validation_labels)
        positive_count = sum(validation_labels.values())
        sampled = _candidate_records(args.sampled_candidates)
        candidates = _candidate_records(args.input)
        supplemental = []
        if positive_count < args.minimum_validation_positives:
            remaining_budget = max(0, args.maximum_total_labels - len(sampled))
            supplemental = supplemental_audit_sample(
                candidates,
                excluded_candidate_ids=(
                    candidate.candidate_id for candidate in sampled
                ),
                maximum_size=remaining_budget,
                seed=args.seed,
            )
        _ensure_parent(args.output)
        write_jsonl(args.output, (row.to_dict() for row in supplemental))
        _write_json(
            f"{args.output}.manifest.json",
            {
                "validation_positive_count": positive_count,
                "trigger_threshold": args.minimum_validation_positives,
                "initial_sample_count": len(sampled),
                "supplemental_sample_count": len(supplemental),
                "maximum_total_labels": args.maximum_total_labels,
                "candidate_ids": [row.candidate_id for row in supplemental],
            },
        )

    def _trajectory_sample(self, args):
        _require(args, "input", "output")
        selected = sample_trajectory_audit(
            read_jsonl(args.input),
            sample_size=args.trajectory_sample_size,
            seed=args.seed,
        )
        _ensure_parent(args.output)
        write_jsonl(args.output, selected)

    def _frozen_report(self, args):
        """Evaluate extra audit labels without exposing threshold selection."""

        _require(args, "input", "validation_labels", "threshold_output", "output")
        labels, label_rows = _successful_labels(args.validation_labels)
        examples = _weighted_examples(_candidate_records(args.input), labels)
        if not examples:
            raise ValueError("No successful annotations are available for audit.")
        artifact = ThresholdArtifact.load(args.threshold_output)
        report = validation_report(
            examples,
            artifact.entailment_threshold,
            artifact.contradiction_threshold,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        report.update(
            {
                "annotation_failures": len(label_rows) - len(labels),
                "threshold_changed": False,
                "threshold_file": args.threshold_output,
            }
        )
        _write_json(args.output, report)

    def _trajectory_report(self, args):
        _require(args, "input", "validation_labels", "threshold_output", "output")
        labels, label_rows = _successful_labels(args.validation_labels)
        artifact = ThresholdArtifact.load(args.threshold_output)
        report = audit_fixed_threshold(
            read_jsonl(args.input),
            labels,
            entailment_threshold=artifact.entailment_threshold,
            contradiction_threshold=artifact.contradiction_threshold,
        )
        report["annotation_failures"] = len(label_rows) - len(labels)
        report["judge_backend"] = artifact.judge_backend
        report["judge_model"] = artifact.judge_model
        if args.base_candidates:
            base_rows = read_jsonl(args.base_candidates)
            trajectory_rows = read_jsonl(args.input)
            report["score_distribution_shift"] = distribution_shift_report(
                base_rows,
                trajectory_rows,
                entailment_threshold=artifact.entailment_threshold,
                contradiction_threshold=artifact.contradiction_threshold,
            )
        if args.base_candidates and args.base_labels:
            base_labels, base_label_rows = _successful_labels(args.base_labels)
            base_examples = _weighted_examples(
                _candidate_records(args.base_candidates), base_labels
            )
            base_metrics = validation_report(
                base_examples,
                artifact.entailment_threshold,
                artifact.contradiction_threshold,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            report["base_frozen_threshold_metrics"] = base_metrics
            report["base_annotation_failures"] = (
                len(base_label_rows) - len(base_labels)
            )
        _write_json(args.output, report)

    def _judge_agreement(self, args):
        _require(args, "left_labels", "right_labels", "output")
        left, _ = _successful_labels(args.left_labels)
        right, _ = _successful_labels(args.right_labels)
        _write_json(args.output, judge_agreement(left, right))

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "semdt-calibrate",
            help="run a resumable SemDT calibration stage",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--stage",
            required=True,
            choices=[
                "nli-score",
                "freeze-split",
                "annotate",
                "tune-validate",
                "supplemental-audit",
                "frozen-report",
                "trajectory-sample",
                "trajectory-report",
                "judge-agreement",
            ],
        )
        parser.add_argument("--input")
        parser.add_argument("--output")
        parser.add_argument("--output-dir")
        parser.add_argument("--profile-output")
        parser.add_argument("--dataset")
        parser.add_argument("--task-definition")
        parser.add_argument("--judge-config")
        parser.add_argument("--annotation-batch-size", type=int, default=32)
        parser.add_argument("--limit", type=int)
        parser.add_argument("--seed", type=int, default=765)
        parser.add_argument("--sample-size", type=int, default=1000)
        parser.add_argument("--search-size", type=int, default=800)
        parser.add_argument("--candidate-batch-size", type=int, default=1024)
        parser.add_argument(
            "--nli-model", default="FacebookAI/roberta-large-mnli"
        )
        parser.add_argument("--nli-model-revision")
        parser.add_argument("--nli-tokenizer-revision")
        parser.add_argument("--nli-device")
        parser.add_argument(
            "--nli-dtype",
            choices=["float32", "float16", "bfloat16"],
            default="float32",
        )
        parser.add_argument("--nli-batch-size", type=int, default=32)
        parser.add_argument("--nli-max-length", type=int, default=512)
        parser.add_argument(
            "--nli-truncation-strategy",
            choices=["longest_first", "only_first", "only_second"],
            default="longest_first",
        )
        parser.add_argument("--search-candidates")
        parser.add_argument("--validation-candidates")
        parser.add_argument("--sampled-candidates")
        parser.add_argument("--search-labels")
        parser.add_argument("--validation-labels")
        parser.add_argument("--base-candidates")
        parser.add_argument("--base-labels")
        parser.add_argument("--left-labels")
        parser.add_argument("--right-labels")
        parser.add_argument("--split-manifest")
        parser.add_argument("--threshold-output")
        parser.add_argument("--report-output")
        # The formal orchestrator always passes this from dataset YAML.
        parser.add_argument(
            "--threshold-search-method",
            choices=["grid"],
        )
        parser.add_argument("--min-precision", type=float, default=0.95)
        parser.add_argument("--threshold-step", type=float, default=0.01)
        parser.add_argument("--bootstrap-samples", type=int, default=10000)
        parser.add_argument("--minimum-validation-positives", type=int, default=100)
        parser.add_argument("--maximum-total-labels", type=int, default=2000)
        parser.add_argument("--trajectory-sample-size", type=int, default=100)
        parser.set_defaults(func=SemDTCalibrationCommand())
        return parser
