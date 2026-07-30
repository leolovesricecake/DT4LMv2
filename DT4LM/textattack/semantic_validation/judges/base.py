"""Shared prompt, configuration, and parsing for semantic judges."""

from dataclasses import dataclass, fields
import hashlib
import json
from typing import Any, Dict, Protocol, Sequence

import yaml

from ..schemas import JudgeExample, JudgeResult


PROMPT_VERSION = "semdt-semantic-preservation-v1"
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"semantic_preserved": {"type": "boolean"}},
    "required": ["semantic_preserved"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class JudgeConfig:
    """One backend configuration with a redacted representation."""

    backend: str
    model: str
    base_url: str = None
    api_key: str = None
    revision: str = None
    timeout: float = 60.0
    max_retries: int = 3
    device: str = None
    dtype: str = "float16"
    batch_size: int = 1
    max_new_tokens: int = 32

    def __repr__(self):
        values = []
        for item in fields(self):
            value = "***" if item.name == "api_key" and self.api_key else getattr(self, item.name)
            values.append(f"{item.name}={value!r}")
        return f"JudgeConfig({', '.join(values)})"


class SemanticJudge(Protocol):
    """Backend-neutral annotation protocol."""

    def annotate(self, examples: Sequence[JudgeExample]) -> list[JudgeResult]:
        raise NotImplementedError()


def load_judge_config(path: str) -> JudgeConfig:
    """Load one selected backend from a local YAML configuration."""

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    backend = raw.get("backend")
    if backend not in {"openai", "hf"}:
        raise ValueError("Judge backend must be either 'openai' or 'hf'.")
    backend_config = raw.get(backend, {})
    config = JudgeConfig(backend=backend, **backend_config)
    if backend == "openai" and not config.api_key:
        raise ValueError("The OpenAI-compatible judge requires a non-empty API key.")
    return config


def build_semantic_prompt(example: JudgeExample) -> str:
    """Render the only question that both judge backends may answer."""

    payload = {
        "task_definition": example.task_definition,
        "dataset": example.dataset,
        "ground_truth_label": example.ground_truth_label,
        "label_name": example.label_name,
        "original_input": example.original_fields,
        "candidate_input": example.candidate_fields,
        "question": (
            "Does the candidate preserve the meaning of the original input "
            "and still correspond to the original ground-truth label?"
        ),
    }
    return (
        f"Prompt version: {PROMPT_VERSION}\n"
        "Judge only the supplied example. Return exactly one JSON object that "
        "matches the required boolean schema.\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def parse_boolean_response(text: str) -> bool:
    """Strictly accept the one-field boolean schema and nothing else."""

    parsed = json.loads(text)
    if (
        not isinstance(parsed, dict)
        or set(parsed) != {"semantic_preserved"}
        or type(parsed["semantic_preserved"]) is not bool
    ):
        raise ValueError("Judge output does not match the strict boolean schema.")
    return parsed["semantic_preserved"]


def public_config(config: JudgeConfig) -> Dict[str, Any]:
    """Return loggable settings while deliberately omitting the API key."""

    return {
        item.name: getattr(config, item.name)
        for item in fields(config)
        if item.name != "api_key"
    }
