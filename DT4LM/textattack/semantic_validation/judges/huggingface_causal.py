"""Local Hugging Face causal-language-model semantic judge."""

from datetime import datetime, timezone
import time
from typing import Sequence
import warnings

from .base import (
    JudgeConfig,
    build_semantic_prompt,
    parse_boolean_response,
    prompt_hash,
)
from ..schemas import JudgeExample, JudgeResult


class HuggingFaceCausalLMJudge:
    """Deterministically annotate examples with a local instruction model."""

    backend = "hf"

    def __init__(self, config: JudgeConfig, model=None, tokenizer=None):
        if config.backend != self.backend:
            raise ValueError("HuggingFaceCausalLMJudge requires backend='hf'.")
        self.config = config
        import torch

        if config.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"Unsupported HF judge dtype {config.dtype!r}.")
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.requested_dtype = config.dtype
        self.dtype = self._resolve_dtype(torch, config.dtype)
        loaded_model = model is None
        if model is None or tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                config.model, revision=config.revision
            )
            model = AutoModelForCausalLM.from_pretrained(
                config.model,
                revision=config.revision,
                torch_dtype=self.dtype,
            )
        self.model = model
        self.tokenizer = tokenizer
        # Decoder-only batching requires a real pad token and left padding so
        # every continuation starts after the final prompt token.
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("HF judge tokenizer needs a pad or EOS token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        if loaded_model:
            self.model.to(device=self.device, dtype=self.dtype)
        else:
            self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        tokenizer_init = getattr(self.tokenizer, "init_kwargs", {}) or {}
        self.resolved_model_revision = (
            getattr(self.model.config, "_commit_hash", None)
            or self.config.revision
        )
        self.resolved_tokenizer_revision = (
            getattr(self.tokenizer, "_commit_hash", None)
            or tokenizer_init.get("_commit_hash")
            or self.config.revision
        )

    def _resolve_dtype(self, torch, requested):
        """Choose a device-supported dtype and make every fallback explicit."""

        if self.device.type == "cpu" and requested != "float32":
            warnings.warn(
                f"HF judge dtype {requested} is unsupported on CPU; "
                "falling back to float32.",
                RuntimeWarning,
            )
            return torch.float32
        if requested == "bfloat16":
            if self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
                warnings.warn(
                    "CUDA does not support bfloat16; falling back to float16.",
                    RuntimeWarning,
                )
                return torch.float16
            if self.device.type not in {"cpu", "cuda"}:
                warnings.warn(
                    f"bfloat16 is not enabled for {self.device.type}; "
                    "falling back to float16.",
                    RuntimeWarning,
                )
                return torch.float16
        return getattr(torch, requested)

    def _metadata(self):
        """Return reproducibility settings without any local secret."""

        return {
            "revision": self.config.revision,
            "resolved_model_revision": self.resolved_model_revision,
            "resolved_tokenizer_revision": self.resolved_tokenizer_revision,
            "requested_dtype": self.requested_dtype,
            "effective_dtype": str(self.dtype).replace("torch.", ""),
            "device": str(self.device),
        }

    def annotate(self, examples: Sequence[JudgeExample]) -> list[JudgeResult]:
        results = []
        batch_size = max(1, self.config.batch_size)
        for start in range(0, len(examples), batch_size):
            results.extend(self._annotate_batch(examples[start : start + batch_size]))
        return results

    def _annotate_batch(self, examples: Sequence[JudgeExample]):
        import torch

        prompts = [build_semantic_prompt(example) for example in examples]
        rendered = [
            self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "Return only the requested JSON object.",
                    },
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        started = [time.perf_counter() for _ in examples]
        encoded = self.tokenizer(rendered, padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        texts = self.tokenizer.batch_decode(
            generated[:, prompt_width:], skip_special_tokens=True
        )
        return [
            self._parse_with_repairs(example, prompt, text, began)
            for example, prompt, text, began in zip(
                examples, prompts, texts, started
            )
        ]

    def _parse_with_repairs(self, example, prompt, initial_text, started_at):
        text = initial_text
        last_error = None
        for attempt in range(1, self.config.max_retries + 2):
            try:
                value = parse_boolean_response(text.strip())
                return JudgeResult(
                    candidate_id=example.candidate_id,
                    semantic_preserved=value,
                    backend=self.backend,
                    model=self.config.model,
                    prompt_hash=prompt_hash(prompt),
                    success=True,
                    attempts=attempt,
                    latency_seconds=time.perf_counter() - started_at,
                    prompt_text=prompt,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    metadata=self._metadata(),
                )
            except Exception as exc:
                last_error = exc
                if attempt <= self.config.max_retries:
                    # Repair retries preserve the semantic question and ask only
                    # for schema correction, so no new annotation criterion leaks in.
                    text = self._repair_once(prompt)
        return JudgeResult(
            candidate_id=example.candidate_id,
            semantic_preserved=None,
            backend=self.backend,
            model=self.config.model,
            prompt_hash=prompt_hash(prompt),
            success=False,
            attempts=self.config.max_retries + 1,
            latency_seconds=time.perf_counter() - started_at,
            prompt_text=prompt,
            created_at=datetime.now(timezone.utc).isoformat(),
            error_type=type(last_error).__name__,
            error_message=str(last_error),
            metadata=self._metadata(),
        )

    def _repair_once(self, prompt: str) -> str:
        """Run one deterministic format-repair generation."""

        import torch

        rendered = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "The previous output was invalid. Return only "
                        '{"semantic_preserved": true} or '
                        '{"semantic_preserved": false}.'
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(rendered, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
