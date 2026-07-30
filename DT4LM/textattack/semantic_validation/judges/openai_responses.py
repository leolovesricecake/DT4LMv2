"""OpenAI Responses-compatible semantic judge backend."""

from datetime import datetime, timezone
from importlib.metadata import version
import time
from typing import Any, Sequence

from .base import (
    OUTPUT_SCHEMA,
    JudgeConfig,
    build_semantic_prompt,
    parse_boolean_response,
    prompt_hash,
)
from ..schemas import JudgeExample, JudgeResult


def _get(value: Any, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class OpenAIResponsesJudge:
    """Annotate candidates through a configurable Responses API endpoint."""

    backend = "openai"

    def __init__(self, config: JudgeConfig, client=None):
        if config.backend != self.backend:
            raise ValueError("OpenAIResponsesJudge requires backend='openai'.")
        if not config.api_key:
            raise ValueError("A non-empty API key is required.")
        self.config = config
        if client is None:
            # Import lazily so local-HF experiments do not require the SDK.
            from openai import OpenAI

            client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
                max_retries=0,
            )
        self.client = client

    def annotate(self, examples: Sequence[JudgeExample]) -> list[JudgeResult]:
        return [self._annotate_one(example) for example in examples]

    def _annotate_one(self, example: JudgeExample) -> JudgeResult:
        prompt = build_semantic_prompt(example)
        hashed_prompt = prompt_hash(prompt)
        started_at = time.perf_counter()
        last_error = None
        for attempt in range(1, self.config.max_retries + 2):
            try:
                response = self.client.responses.create(
                    model=self.config.model,
                    input=[
                        {
                            "role": "system",
                            "content": (
                                "You are a deterministic semantic-preservation "
                                "annotator."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_output_tokens=self.config.max_new_tokens,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "semantic_judgment",
                            "strict": True,
                            "schema": OUTPUT_SCHEMA,
                        }
                    },
                )
                if _get(response, "status") == "incomplete":
                    reason = _get(_get(response, "incomplete_details", {}), "reason")
                    raise RuntimeError(f"Incomplete response: {reason}")
                text = self._extract_output_text(response)
                semantic_preserved = parse_boolean_response(text)
                usage = _get(response, "usage", {})
                return JudgeResult(
                    candidate_id=example.candidate_id,
                    semantic_preserved=semantic_preserved,
                    backend=self.backend,
                    model=self.config.model,
                    prompt_hash=hashed_prompt,
                    success=True,
                    attempts=attempt,
                    latency_seconds=time.perf_counter() - started_at,
                    prompt_text=prompt,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    request_id=_get(response, "id"),
                    input_tokens=_get(usage, "input_tokens"),
                    output_tokens=_get(usage, "output_tokens"),
                    metadata={
                        "base_url": self.config.base_url,
                        "sdk_version": version("openai"),
                    },
                )
            except Exception as exc:
                last_error = exc

        # Error text is retained for diagnosis, but any accidental occurrence
        # of the configured credential is redacted before serialization.
        message = str(last_error)
        if self.config.api_key:
            message = message.replace(self.config.api_key, "***")
        return JudgeResult(
            candidate_id=example.candidate_id,
            semantic_preserved=None,
            backend=self.backend,
            model=self.config.model,
            prompt_hash=hashed_prompt,
            success=False,
            attempts=self.config.max_retries + 1,
            latency_seconds=time.perf_counter() - started_at,
            prompt_text=prompt,
            created_at=datetime.now(timezone.utc).isoformat(),
            error_type=type(last_error).__name__,
            error_message=message,
            metadata={
                "base_url": self.config.base_url,
                "sdk_version": version("openai"),
            },
        )

    @staticmethod
    def _extract_output_text(response) -> str:
        """Distinguish refusals and missing content from valid output text."""

        for item in _get(response, "output", []) or []:
            if _get(item, "type") != "message":
                continue
            for content in _get(item, "content", []) or []:
                content_type = _get(content, "type")
                if content_type == "refusal":
                    raise RuntimeError("Judge refused the semantic annotation.")
                if content_type == "output_text":
                    return _get(content, "text", "")
        output_text = _get(response, "output_text")
        if output_text:
            return output_text
        raise RuntimeError("Responses API returned no output text.")
