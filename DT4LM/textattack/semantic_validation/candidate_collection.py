"""Candidate observation, deduplication, stratification, and fixed splitting."""

from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .schemas import CandidateRecord, read_jsonl, write_jsonl


def _stable_candidate_id(dataset_index: int, fields: Mapping[str, str]) -> str:
    payload = json.dumps(
        [dataset_index, list(fields.items())],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CandidateObserver:
    """Append candidates after query-budget truncation without changing search."""

    def __init__(self, path: str, metadata: Dict[str, Any] = None):
        self.path = path
        self.metadata = dict(metadata or {})
        self._context = None
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        # Loading existing IDs makes interrupted collection idempotent.
        self._seen = {
            row["candidate_id"]
            for row in read_jsonl(path)
        } if os.path.exists(path) else set()

    def start_example(self, dataset_index, original_text, ground_truth_output):
        """Set immutable context before the original input model query."""

        label_names = original_text.attack_attrs.get("label_names") or []
        label = int(ground_truth_output)
        label_name = str(label_names[label]) if label < len(label_names) else str(label)
        self._context = {
            "dataset_index": int(dataset_index),
            "original_fields": dict(original_text.text_input),
            "ground_truth_label": label,
            "label_name": label_name,
            "original_text": original_text,
        }

    def observe(self, candidates, *, num_queries_before, query_budget):
        if self._context is None:
            raise RuntimeError("CandidateObserver.start_example must be called first.")
        rows = []
        for offset, candidate in enumerate(candidates, start=1):
            candidate_fields = dict(candidate.text_input)
            candidate_id = _stable_candidate_id(
                self._context["dataset_index"], candidate_fields
            )
            if candidate_id in self._seen:
                continue
            changed_fields = [
                field
                for field, value in candidate_fields.items()
                if self._context["original_fields"].get(field) != value
            ]
            row = CandidateRecord(
                candidate_id=candidate_id,
                dataset=str(self.metadata.get("dataset", "")),
                split=str(self.metadata.get("split", "train")),
                dataset_index=self._context["dataset_index"],
                ground_truth_label=self._context["ground_truth_label"],
                label_name=self._context["label_name"],
                original_fields=dict(self._context["original_fields"]),
                candidate_fields=candidate_fields,
                changed_fields=changed_fields,
                modified_indices=sorted(
                    candidate.attack_attrs.get("modified_indices", set())
                ),
                modification_cost=candidate.modification_rate(
                    self._context["original_text"]
                ),
                search_round=int(candidate.attack_attrs.get("search_round", 0)),
                candidate_order=int(
                    candidate.attack_attrs.get("candidate_order", offset - 1)
                ),
                model_pair_query=int(num_queries_before + offset),
                metadata={
                    **self.metadata,
                    "query_budget": (
                        None if query_budget == float("inf") else query_budget
                    ),
                },
            )
            rows.append(row.to_dict())
            self._seen.add(candidate_id)

        if rows:
            with open(self.path, "a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(
                        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
                    )


def score_candidate_records(
    candidates: Sequence[CandidateRecord],
    nli,
    *,
    candidate_batch_size: int = 1024,
) -> List[CandidateRecord]:
    """Offline-score structured candidates with the online NLI implementation."""

    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive.")
    scored = []
    for start in range(0, len(candidates), candidate_batch_size):
        batch = candidates[start : start + candidate_batch_size]
        field_pairs = []
        layouts = []
        for candidate in batch:
            changed = [
                field
                for field in candidate.changed_fields
                if candidate.original_fields.get(field)
                != candidate.candidate_fields.get(field)
            ]
            begin = len(field_pairs)
            field_pairs.extend(
                (
                    candidate.original_fields[field],
                    candidate.candidate_fields[field],
                )
                for field in changed
            )
            layouts.append((changed, begin, len(field_pairs)))
        field_scores = nli.score_field_pairs(field_pairs)
        nli.profile.candidates += len(batch)
        for candidate, (changed, begin, end) in zip(batch, layouts):
            selected_scores = field_scores[begin:end]
            entailment = min(
                (score.entailment_score for score in selected_scores),
                default=1.0,
            )
            contradiction = max(
                (score.contradiction_score for score in selected_scores),
                default=0.0,
            )
            truncated = any(
                score.forward_truncated or score.reverse_truncated
                for score in selected_scores
            )
            if truncated:
                nli.profile.truncated_candidates += 1
            metadata = {
                **candidate.metadata,
                "nli_config": {
                    "model_name_or_path": nli.model_name_or_path,
                    "model_revision": nli.model_revision,
                    "tokenizer_revision": nli.tokenizer_revision,
                    "resolved_model_revision": nli.resolved_model_revision,
                    "resolved_tokenizer_revision": (
                        nli.resolved_tokenizer_revision
                    ),
                    "max_length": nli.max_length,
                    "truncation_strategy": nli.truncation_strategy,
                    "requested_dtype": nli.requested_dtype,
                    "effective_dtype": str(nli.dtype).replace("torch.", ""),
                },
                "nli_field_scores": {
                    field: asdict(score)
                    for field, score in zip(changed, selected_scores)
                },
                "nli_truncated": truncated,
            }
            scored.append(
                replace(
                    candidate,
                    entailment_score=entailment,
                    contradiction_score=contradiction,
                    metadata=metadata,
                )
            )
    return scored


def score_stratum(entailment_score: float, contradiction_score: float) -> str:
    """Map NLI scores to a deterministic two-dimensional decile cell."""

    entailment_decile = min(9, max(0, int(entailment_score * 10)))
    contradiction_decile = min(9, max(0, int(contradiction_score * 10)))
    return f"e{entailment_decile}-c{contradiction_decile}"


def _allocate_by_largest_remainder(
    group_sizes: Mapping[Any, int], total: int
) -> Dict[Any, int]:
    """Allocate an exact total proportionally while respecting group capacity."""

    population = sum(group_sizes.values())
    if total < 0 or total > population:
        raise ValueError(
            f"Cannot allocate {total} samples from a population of {population}."
        )
    if population == 0:
        return {}
    quotas = {
        group: total * size / population for group, size in group_sizes.items()
    }
    allocation = {
        group: min(size, int(math.floor(quotas[group])))
        for group, size in group_sizes.items()
    }
    remaining = total - sum(allocation.values())
    ranking = sorted(
        group_sizes,
        key=lambda group: (
            -(quotas[group] - math.floor(quotas[group])),
            str(group),
        ),
    )
    while remaining:
        made_progress = False
        for group in ranking:
            if allocation[group] < group_sizes[group]:
                allocation[group] += 1
                remaining -= 1
                made_progress = True
                if remaining == 0:
                    break
        if not made_progress:
            raise RuntimeError("Largest-remainder allocation could not finish.")
    return allocation


def stratified_candidate_sample(
    candidates: Sequence[CandidateRecord],
    *,
    sample_size: int = 1000,
    seed: int = 765,
) -> List[CandidateRecord]:
    """Sample score-grid strata and attach inverse inclusion weights."""

    if len(candidates) < sample_size:
        raise ValueError(
            f"Need {sample_size} scored candidates, received {len(candidates)}."
        )
    groups = defaultdict(list)
    for candidate in candidates:
        if (
            candidate.entailment_score is None
            or candidate.contradiction_score is None
        ):
            raise ValueError("Candidates must be NLI-scored before stratification.")
        stratum = score_stratum(
            candidate.entailment_score, candidate.contradiction_score
        )
        groups[(candidate.ground_truth_label, stratum)].append(candidate)

    if sample_size < len(groups):
        raise ValueError(
            f"Sample size {sample_size} cannot cover all {len(groups)} strata."
        )
    # Reserve one example per non-empty cell, then distribute the remaining
    # budget proportionally. This guarantees score-boundary coverage.
    allocation = {group: 1 for group in groups}
    extra_capacity = {
        group: len(rows) - 1 for group, rows in groups.items()
    }
    extra = _allocate_by_largest_remainder(
        extra_capacity, sample_size - len(groups)
    )
    allocation = {
        group: allocation[group] + extra.get(group, 0) for group in groups
    }
    rng = random.Random(seed)
    selected = []
    for group in sorted(groups, key=str):
        rows = sorted(groups[group], key=lambda row: row.candidate_id)
        count = allocation[group]
        sampled = rng.sample(rows, count)
        weight = len(rows) / count
        selected.extend(
            replace(row, stratum=group[1], inclusion_weight=weight)
            for row in sampled
        )
    # A final stable order ensures both judges see exactly the same examples.
    return sorted(selected, key=lambda row: row.candidate_id)


def freeze_calibration_split(
    candidates: Sequence[CandidateRecord],
    *,
    search_size: int = 800,
    seed: int = 765,
) -> Tuple[List[CandidateRecord], List[CandidateRecord]]:
    """Freeze one stratified search/validation split before judge calls."""

    if not 0 < search_size < len(candidates):
        raise ValueError("search_size must leave a non-empty validation set.")
    groups = defaultdict(list)
    for candidate in candidates:
        if not candidate.stratum:
            raise ValueError("Every sampled candidate must have a stratum.")
        groups[(candidate.ground_truth_label, candidate.stratum)].append(candidate)
    allocation = _allocate_by_largest_remainder(
        {group: len(rows) for group, rows in groups.items()}, search_size
    )
    rng = random.Random(seed)
    search, validation = [], []
    for group in sorted(groups, key=str):
        rows = sorted(groups[group], key=lambda row: row.candidate_id)
        rng.shuffle(rows)
        count = allocation[group]
        # Initial weights recover the full stratum from 1000 samples. The
        # independent 800/200 sets have another inclusion probability and need
        # their own full-stratum inverse weights.
        population = sum(float(row.inclusion_weight or 1.0) for row in rows)
        search.extend(
            replace(row, inclusion_weight=population / count)
            for row in rows[:count]
        )
        validation_count = len(rows) - count
        validation.extend(
            replace(row, inclusion_weight=population / validation_count)
            for row in rows[count:]
        )
    return (
        sorted(search, key=lambda row: row.candidate_id),
        sorted(validation, key=lambda row: row.candidate_id),
    )


def write_split_manifest(
    path: str,
    search: Iterable[CandidateRecord],
    validation: Iterable[CandidateRecord],
    *,
    seed: int,
) -> None:
    """Write candidate IDs only so annotation cannot mutate the split."""

    search = list(search)
    validation = list(validation)
    payload = {
        "seed": seed,
        "search_ids": [row.candidate_id for row in search],
        "validation_ids": [row.candidate_id for row in validation],
    }
    # Counts make the second-stage inclusion weights independently auditable.
    strata = defaultdict(lambda: {"search": 0, "validation": 0})
    for name, rows in (("search", search), ("validation", validation)):
        for row in rows:
            key = f"{row.ground_truth_label}:{row.stratum}"
            strata[key][name] += 1
    payload["strata_counts"] = dict(sorted(strata.items()))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)


def supplemental_audit_sample(
    candidates: Sequence[CandidateRecord],
    *,
    excluded_candidate_ids: Iterable[str],
    maximum_size: int = 1000,
    seed: int = 765,
) -> List[CandidateRecord]:
    """Draw an independent weighted sample without touching the frozen split."""

    if maximum_size < 0:
        raise ValueError("maximum_size cannot be negative.")
    excluded = set(excluded_candidate_ids)
    remaining = [
        candidate
        for candidate in candidates
        if candidate.candidate_id not in excluded
    ]
    if not remaining or maximum_size == 0:
        return []
    sample_size = min(maximum_size, len(remaining))
    return stratified_candidate_sample(
        remaining, sample_size=sample_size, seed=seed
    )
