"""Structured per-sample JSONL logging for reproducible experiments."""

import json

from textattack.shared import logger

from .logger import Logger


def _serialize_score(score):
    if hasattr(score, "to_serializable"):
        return score.to_serializable()
    if hasattr(score, "item"):
        return score.item()
    return score


class JSONLLogger(Logger):
    """Write one complete, machine-readable attack result per line."""

    def __init__(self, filename="results.jsonl"):
        logger.info(f"Logging structured results to JSONL at path {filename}")
        self.filename = filename
        self._rows = []
        self._flushed = True

    def log_attack_result(self, result):
        original = result.original_result
        perturbed = result.perturbed_result
        attrs = perturbed.attacked_text.attack_attrs
        row = {
            "dataset_index": attrs.get(
                "dataset_index",
                original.attacked_text.attack_attrs.get("dataset_index"),
            ),
            "run_config_hash": attrs.get(
                "run_config_hash",
                original.attacked_text.attack_attrs.get("run_config_hash"),
            ),
            "original_input": dict(original.attacked_text.text_input),
            "candidate_input": dict(perturbed.attacked_text.text_input),
            "ground_truth_output": original.ground_truth_output,
            "result_type": result.__class__.__name__.replace("AttackResult", ""),
            "success": result.__class__.__name__ == "SuccessfulAttackResult",
            "model_pair_queries": result.num_queries,
            "objective": getattr(perturbed, "objective_name", None),
            "objective_score": _serialize_score(perturbed.score),
            "new_model_output": getattr(perturbed, "new_model_output", None),
            "old_model_output": getattr(perturbed, "old_model_output", None),
            "modified_indices": sorted(attrs.get("modified_indices", set())),
            "modification_rate": perturbed.attacked_text.modification_rate(
                original.attacked_text
            ),
            "nli": attrs.get("nli"),
            "nli_profile": getattr(result, "nli_profile", None),
            "wall_clock_seconds": getattr(result, "wall_clock_seconds", None),
            "peak_vram_bytes": getattr(result, "peak_vram_bytes", None),
        }
        self._rows.append(row)
        self._flushed = False

    def flush(self):
        with open(self.filename, "w", encoding="utf-8") as handle:
            for row in self._rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        self._flushed = True

    def close(self):
        super().close()

    def __del__(self):
        if not self._flushed:
            logger.warning("JSONLLogger exiting without calling flush().")
