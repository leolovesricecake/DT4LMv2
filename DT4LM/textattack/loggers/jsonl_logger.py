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


def _result_status(result):
    """Map TextAttack result classes onto the frozen three-state protocol."""

    class_name = result.__class__.__name__
    if class_name == "SuccessfulAttackResult":
        return "successful"
    if class_name == "SkippedAttackResult":
        return "skipped"
    return "failed"


def _initial_state(original):
    """Classify the original pair prediction before any perturbation search."""

    new_output = getattr(original, "new_model_output", None) or {}
    old_output = getattr(original, "old_model_output", None) or {}
    label = int(original.ground_truth_output)
    new_correct = new_output.get("predicted_label") == label
    old_correct = old_output.get("predicted_label") == label
    if new_correct and old_correct:
        return "both_correct"
    if new_correct:
        return "new_correct_old_wrong"
    if old_correct:
        return "already_differential"
    return "both_wrong"


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
        status = _result_status(result)
        initial_state = _initial_state(original)
        initial_queries = int(getattr(original, "num_queries", 1) or 1)
        row = {
            "schema_version": 3,
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
            "result_status": status,
            "initial_state": initial_state,
            "skip_reason": (
                "already_differential" if status == "skipped" else None
            ),
            # Keep the legacy boolean during migration; all new metrics use
            # result_status so skipped and failed cannot be conflated.
            "success": status == "successful",
            "model_pair_queries": result.num_queries,
            "initial_model_pair_queries": initial_queries,
            "search_model_pair_queries": max(0, result.num_queries - initial_queries),
            "queries_to_success": (
                result.num_queries if status == "successful" else None
            ),
            "objective": getattr(perturbed, "objective_name", None),
            "objective_score": _serialize_score(perturbed.score),
            "original_new_model_output": getattr(
                original, "new_model_output", None
            ),
            "original_old_model_output": getattr(
                original, "old_model_output", None
            ),
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
            "search_diagnostics": getattr(
                perturbed, "search_diagnostics", None
            ),
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
