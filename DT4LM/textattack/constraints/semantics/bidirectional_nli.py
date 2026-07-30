"""Batched bidirectional NLI constraint for semantic preservation."""

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from textattack.constraints import Constraint
from textattack.shared import logger


@dataclass(frozen=True)
class NLIFieldScore:
    """Two-direction NLI probabilities for one changed input field."""

    forward_entailment: float
    reverse_entailment: float
    forward_contradiction: float
    reverse_contradiction: float
    forward_truncated: bool = False
    reverse_truncated: bool = False

    @property
    def entailment_score(self) -> float:
        return min(self.forward_entailment, self.reverse_entailment)

    @property
    def contradiction_score(self) -> float:
        return max(self.forward_contradiction, self.reverse_contradiction)


@dataclass
class NLIProfile:
    """Counters required to explain SemDT's non-victim-model cost."""

    candidates: int = 0
    logical_directional_pairs: int = 0
    directional_pairs: int = 0
    batches: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    truncated_directional_pairs: int = 0
    truncated_candidates: int = 0
    inference_seconds: float = 0.0
    peak_vram_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        lookups = self.cache_hits + self.cache_misses
        data["cache_hit_rate"] = self.cache_hits / lookups if lookups else None
        data["seconds_per_candidate"] = (
            self.inference_seconds / self.candidates if self.candidates else None
        )
        data["truncated_directional_pair_rate"] = (
            self.truncated_directional_pairs / self.directional_pairs
            if self.directional_pairs
            else None
        )
        data["truncated_candidate_rate"] = (
            self.truncated_candidates / self.candidates
            if self.candidates
            else None
        )
        return data


def _normalized_label_indices(id2label: Mapping) -> Tuple[int, int]:
    """Resolve labels by name so model-specific class ordering is harmless."""

    entailment = contradiction = None
    for raw_index, raw_name in id2label.items():
        name = str(raw_name).strip().lower()
        if "entail" in name:
            entailment = int(raw_index)
        if "contrad" in name:
            contradiction = int(raw_index)
    if entailment is None or contradiction is None:
        raise ValueError(
            "NLI model config.id2label must name entailment and contradiction."
        )
    return entailment, contradiction


class BidirectionalNLI(Constraint):
    """Require mutual entailment and low contradiction for changed fields."""

    def __init__(
        self,
        model_name_or_path="FacebookAI/roberta-large-mnli",
        *,
        model_revision=None,
        tokenizer_revision=None,
        entailment_threshold=0.90,
        contradiction_threshold=0.05,
        device=None,
        dtype="float32",
        batch_size=32,
        max_length=512,
        truncation_strategy="longest_first",
        audit_log=None,
        cache_size=2**18,
        model=None,
        tokenizer=None,
        expected_nli_config=None,
    ):
        super().__init__(compare_against_original=True)
        if not 0 <= entailment_threshold <= 1:
            raise ValueError("entailment_threshold must lie in [0, 1].")
        if not 0 <= contradiction_threshold <= 1:
            raise ValueError("contradiction_threshold must lie in [0, 1].")
        if truncation_strategy not in {
            "longest_first",
            "only_first",
            "only_second",
        }:
            raise ValueError("Unsupported NLI truncation strategy.")

        self.model_name_or_path = model_name_or_path
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.entailment_threshold = float(entailment_threshold)
        self.contradiction_threshold = float(contradiction_threshold)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.truncation_strategy = truncation_strategy
        self.audit_log = audit_log
        self.cache_size = int(cache_size)
        if self.batch_size <= 0 or self.max_length <= 0 or self.cache_size <= 0:
            raise ValueError("NLI batch_size, max_length, and cache_size must be positive.")
        self._cache = OrderedDict()
        self.profile = NLIProfile()

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.requested_dtype = dtype
        self.dtype = self._resolve_dtype(dtype)
        loaded_model = model is None
        if model is None or tokenizer is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                revision=tokenizer_revision or model_revision,
                use_fast=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                revision=model_revision,
                torch_dtype=self.dtype,
            )
        self.model = model
        self.tokenizer = tokenizer
        # A model loaded here receives the effective dtype explicitly. Injected
        # test models keep their synthetic dtype and only move to the device.
        if loaded_model:
            self.model.to(device=self.device, dtype=self.dtype)
        else:
            self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.entailment_index, self.contradiction_index = _normalized_label_indices(
            self.model.config.id2label
        )
        self.resolved_model_revision = (
            getattr(self.model.config, "_commit_hash", None) or model_revision
        )
        tokenizer_init = getattr(self.tokenizer, "init_kwargs", {}) or {}
        self.resolved_tokenizer_revision = (
            getattr(self.tokenizer, "_commit_hash", None)
            or tokenizer_init.get("_commit_hash")
            or tokenizer_revision
            or model_revision
        )
        if expected_nli_config:
            resolved = {
                "resolved_model_revision": self.resolved_model_revision,
                "resolved_tokenizer_revision": self.resolved_tokenizer_revision,
            }
            for key, expected_value in expected_nli_config.items():
                if key in resolved and expected_value != resolved[key]:
                    raise ValueError(
                        "Calibrated threshold resolved NLI revision mismatch "
                        f"for {key}: {expected_value!r} != {resolved[key]!r}."
                    )

        if audit_log:
            os.makedirs(os.path.dirname(audit_log) or ".", exist_ok=True)

    @classmethod
    def from_attack_args(cls, args):
        """Build the constraint and optionally load a frozen threshold artifact."""

        entailment = args.nli_entailment_threshold
        contradiction = args.nli_contradiction_threshold
        expected = None
        if args.semantic_threshold_file:
            with open(args.semantic_threshold_file, encoding="utf-8") as handle:
                artifact = json.load(handle)
            expected = artifact.get("nli_config")
            if expected:
                actual = {
                    "model_name_or_path": args.nli_model_name_or_path,
                    "model_revision": args.nli_model_revision,
                    "tokenizer_revision": args.nli_tokenizer_revision,
                    "max_length": args.nli_max_length,
                    "truncation_strategy": args.nli_truncation_strategy,
                }
                for key, value in actual.items():
                    if expected.get(key) != value:
                        raise ValueError(
                            "Calibrated threshold NLI configuration mismatch "
                            f"for {key}: {expected.get(key)!r} != {value!r}."
                        )
            entailment = artifact["entailment_threshold"]
            contradiction = artifact["contradiction_threshold"]
        return cls(
            args.nli_model_name_or_path,
            model_revision=args.nli_model_revision,
            tokenizer_revision=args.nli_tokenizer_revision,
            entailment_threshold=entailment,
            contradiction_threshold=contradiction,
            device=args.nli_device,
            dtype=args.nli_dtype,
            batch_size=args.nli_batch_size,
            max_length=args.nli_max_length,
            truncation_strategy=args.nli_truncation_strategy,
            audit_log=args.nli_audit_log,
            expected_nli_config=expected,
        )

    def _resolve_dtype(self, requested):
        if requested not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"Unsupported NLI dtype {requested!r}.")
        if self.device.type == "cpu" and requested != "float32":
            logger.warning(
                "NLI dtype %s is not used on CPU; falling back to float32.",
                requested,
            )
            return torch.float32
        if requested == "bfloat16" and self.device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                logger.warning(
                    "CUDA device does not support bfloat16; falling back to float16."
                )
                return torch.float16
        return getattr(torch, requested)

    def _cache_key(self, original_text: str, candidate_text: str):
        tokenizer_id = getattr(
            self.tokenizer, "name_or_path", self.model_name_or_path
        )
        return (
            self.model_name_or_path,
            self.model_revision,
            tokenizer_id,
            self.tokenizer_revision,
            self.max_length,
            self.truncation_strategy,
            original_text,
            candidate_text,
        )

    def _cache_get(self, key, *, count_hit=True):
        if key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value
        if count_hit:
            self.profile.cache_hits += 1
        return value

    def _cache_put(self, key, value):
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def score_field_pairs(self, field_pairs):
        """Score ordered original/candidate fields while reusing the NLI cache."""

        field_pairs = list(field_pairs)
        pending = OrderedDict()
        keys = []
        for original, candidate in field_pairs:
            key = self._cache_key(original, candidate)
            keys.append(key)
            if key not in self._cache and key not in pending:
                pending[key] = (original, candidate)
        self.profile.logical_directional_pairs += 2 * len(field_pairs)
        self.profile.cache_misses += len(pending)
        newly_scored_keys = set(pending)
        if pending:
            self._score_pending_fields(pending)
        scores = []
        for key in keys:
            score = self._cache_get(
                key, count_hit=key not in newly_scored_keys
            )
            if score is None:
                raise RuntimeError("Missing NLI score after batched inference.")
            scores.append(score)
        return scores

    def _check_constraint(self, transformed_text, reference_text):
        return bool(
            self._check_constraint_many([transformed_text], reference_text)
        )

    def _check_constraint_many(self, transformed_texts, reference_text):
        if not transformed_texts:
            return []
        reference_fields = dict(reference_text.text_input)
        pending = OrderedDict()
        candidate_fields = []
        for transformed_text in transformed_texts:
            fields = dict(transformed_text.text_input)
            changed = [
                name
                for name, value in fields.items()
                if reference_fields.get(name) != value
            ]
            candidate_fields.append((transformed_text, fields, changed))
            for name in changed:
                key = self._cache_key(reference_fields[name], fields[name])
                if key not in pending and key not in self._cache:
                    pending[key] = (reference_fields[name], fields[name])

        self.profile.candidates += len(transformed_texts)
        self.profile.logical_directional_pairs += 2 * sum(
            len(changed) for _, _, changed in candidate_fields
        )
        self.profile.cache_misses += len(pending)
        newly_scored_keys = set(pending)
        if pending:
            self._score_pending_fields(pending)

        accepted_candidates = []
        for transformed_text, fields, changed in candidate_fields:
            field_scores = {}
            for name in changed:
                key = self._cache_key(reference_fields[name], fields[name])
                score = self._cache_get(
                    key, count_hit=key not in newly_scored_keys
                )
                # Newly inferred items were cache misses before they are read.
                if score is None:
                    raise RuntimeError("Missing NLI score after batched inference.")
                field_scores[name] = score
            # An unchanged candidate should already have been removed by Attack;
            # accepting it here keeps the constraint total if called directly.
            entailment = min(
                (score.entailment_score for score in field_scores.values()),
                default=1.0,
            )
            contradiction = max(
                (score.contradiction_score for score in field_scores.values()),
                default=0.0,
            )
            truncated = any(
                score.forward_truncated or score.reverse_truncated
                for score in field_scores.values()
            )
            if truncated:
                self.profile.truncated_candidates += 1
            accepted = (
                entailment >= self.entailment_threshold
                and contradiction <= self.contradiction_threshold
            )
            transformed_text.attack_attrs["nli"] = {
                "entailment_score": entailment,
                "contradiction_score": contradiction,
                "accepted": accepted,
                "truncated": truncated,
                "changed_fields": changed,
                "field_scores": {
                    name: asdict(score) for name, score in field_scores.items()
                },
            }
            self._write_audit_row(transformed_text, reference_fields)
            if accepted:
                accepted_candidates.append(transformed_text)
        return accepted_candidates

    def _score_pending_fields(self, pending):
        directional_pairs = []
        keys = list(pending)
        for original, candidate in pending.values():
            directional_pairs.extend(((original, candidate), (candidate, original)))
        probabilities, truncated = self._infer_pairs(directional_pairs)
        for index, key in enumerate(keys):
            forward = probabilities[2 * index]
            reverse = probabilities[2 * index + 1]
            score = NLIFieldScore(
                forward_entailment=float(forward[self.entailment_index]),
                reverse_entailment=float(reverse[self.entailment_index]),
                forward_contradiction=float(forward[self.contradiction_index]),
                reverse_contradiction=float(reverse[self.contradiction_index]),
                forward_truncated=truncated[2 * index],
                reverse_truncated=truncated[2 * index + 1],
            )
            self._cache_put(key, score)

    def _infer_pairs(self, directional_pairs: Sequence[Tuple[str, str]]):
        all_probabilities = []
        all_truncated = []
        started_at = time.perf_counter()
        for start in range(0, len(directional_pairs), self.batch_size):
            batch = directional_pairs[start : start + self.batch_size]
            first = [pair[0] for pair in batch]
            second = [pair[1] for pair in batch]
            untruncated = self.tokenizer(
                first,
                second,
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            truncated = [
                len(input_ids) > self.max_length
                for input_ids in untruncated["input_ids"]
            ]
            encoded = self.tokenizer(
                first,
                second,
                add_special_tokens=True,
                padding=True,
                max_length=self.max_length,
                truncation=self.truncation_strategy,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits
                probabilities = torch.softmax(logits.float(), dim=-1)
            all_probabilities.extend(probabilities.cpu().tolist())
            all_truncated.extend(truncated)
            self.profile.batches += 1
        self.profile.directional_pairs += len(directional_pairs)
        self.profile.truncated_directional_pairs += sum(all_truncated)
        self.profile.inference_seconds += time.perf_counter() - started_at
        if self.device.type == "cuda":
            self.profile.peak_vram_bytes = max(
                self.profile.peak_vram_bytes,
                torch.cuda.max_memory_allocated(self.device),
            )
        return all_probabilities, all_truncated

    def _write_audit_row(self, transformed_text, reference_fields):
        if not self.audit_log:
            return
        fields = dict(transformed_text.text_input)
        dataset_index = transformed_text.attack_attrs.get("dataset_index")
        payload = json.dumps(
            [dataset_index, list(fields.items())],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        row = {
            "candidate_id": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "dataset_index": dataset_index,
            "original_fields": dict(reference_fields),
            "candidate_fields": fields,
            "ground_truth_label": transformed_text.attack_attrs.get(
                "ground_truth_output"
            ),
            "label_name": transformed_text.attack_attrs.get(
                "ground_truth_label_name"
            ),
            "search_round": transformed_text.attack_attrs.get("search_round"),
            "candidate_order": transformed_text.attack_attrs.get("candidate_order"),
            **transformed_text.attack_attrs["nli"],
        }
        with open(self.audit_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    def profile_dict(self):
        """Return runtime statistics plus effective reproducibility settings."""

        return {
            **self.profile.to_dict(),
            "model_name_or_path": self.model_name_or_path,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "resolved_model_revision": self.resolved_model_revision,
            "resolved_tokenizer_revision": self.resolved_tokenizer_revision,
            "max_length": self.max_length,
            "truncation_strategy": self.truncation_strategy,
            "requested_dtype": self.requested_dtype,
            "effective_dtype": str(self.dtype).replace("torch.", ""),
            "entailment_threshold": self.entailment_threshold,
            "contradiction_threshold": self.contradiction_threshold,
        }

    def extra_repr_keys(self):
        return [
            "model_name_or_path",
            "entailment_threshold",
            "contradiction_threshold",
            "max_length",
            "truncation_strategy",
        ]
