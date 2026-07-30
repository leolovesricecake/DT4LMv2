"""Serializable schemas shared by semantic-validation pipeline stages."""

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class CandidateRecord:
    """One candidate actually evaluated along the Base or SemDT trajectory."""

    candidate_id: str
    dataset: str
    split: str
    dataset_index: int
    ground_truth_label: int
    label_name: str
    original_fields: Dict[str, str]
    candidate_fields: Dict[str, str]
    changed_fields: List[str]
    modified_indices: List[int]
    modification_cost: float
    search_round: int
    candidate_order: int
    model_pair_query: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    entailment_score: Optional[float] = None
    contradiction_score: Optional[float] = None
    inclusion_weight: Optional[float] = None
    stratum: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        return cls(**dict(data))


@dataclass(frozen=True)
class JudgeExample:
    """The minimal task-aware input presented to either semantic judge."""

    candidate_id: str
    dataset: str
    task_definition: str
    ground_truth_label: int
    label_name: str
    original_fields: Dict[str, str]
    candidate_fields: Dict[str, str]

    @classmethod
    def from_candidate(cls, candidate: CandidateRecord, task_definition: str):
        return cls(
            candidate_id=candidate.candidate_id,
            dataset=candidate.dataset,
            task_definition=task_definition,
            ground_truth_label=candidate.ground_truth_label,
            label_name=candidate.label_name,
            original_fields=dict(candidate.original_fields),
            candidate_fields=dict(candidate.candidate_fields),
        )


@dataclass(frozen=True)
class JudgeResult:
    """One parsed judge response or a retained terminal failure."""

    candidate_id: str
    semantic_preserved: Optional[bool]
    backend: str
    model: str
    prompt_hash: str
    success: bool
    attempts: int
    latency_seconds: float
    prompt_text: Optional[str] = None
    created_at: Optional[str] = None
    request_id: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        return cls(**dict(data))


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read non-empty JSONL rows while reporting the offending line."""

    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}."
                ) from exc
    return rows


def write_jsonl(path: str, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write an immutable pipeline stage as deterministic JSONL."""

    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True))
            handle.write("\n")
