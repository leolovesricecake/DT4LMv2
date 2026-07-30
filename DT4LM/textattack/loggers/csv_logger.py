"""
Attack Logs to CSV
========================
"""

import csv
import json

import pandas as pd

from textattack.shared import AttackedText, logger

from .logger import Logger


class CSVLogger(Logger):
    """Logs attack results to a CSV."""

    def __init__(self, filename="results.csv", color_method="file"):
        logger.info(f"Logging to CSV at path {filename}")
        self.filename = filename
        self.color_method = color_method
        self.row_list = []
        self._flushed = True
        self.df = pd.DataFrame()

    def log_attack_result(self, result):
        original_text, perturbed_text = result.diff_color(self.color_method)
        original_text = original_text.replace("\n", AttackedText.SPLIT_TOKEN)
        perturbed_text = perturbed_text.replace("\n", AttackedText.SPLIT_TOKEN)
        result_type = result.__class__.__name__.replace("AttackResult", "")
        row = {
            "original_text": original_text,
            "perturbed_text": perturbed_text,
            "original_score": result.original_result.score,
            "perturbed_score": result.perturbed_result.score,
            "original_output": result.original_result.output,
            "perturbed_output": result.perturbed_result.output,
            "ground_truth_output": result.original_result.ground_truth_output,
            "num_queries": result.num_queries,
            "result_type": result_type,
            # Keep nested model/objective fields encoded rather than dropping
            # them from the legacy flat CSV artifact.
            "objective": getattr(result.perturbed_result, "objective_name", None),
            "objective_score": json.dumps(
                result.perturbed_result.score.to_serializable()
                if hasattr(result.perturbed_result.score, "to_serializable")
                else result.perturbed_result.score
            ),
            "new_model_output": json.dumps(
                getattr(result.perturbed_result, "new_model_output", None)
            ),
            "old_model_output": json.dumps(
                getattr(result.perturbed_result, "old_model_output", None)
            ),
            "nli": json.dumps(
                result.perturbed_result.attacked_text.attack_attrs.get("nli")
            ),
            "nli_profile": json.dumps(getattr(result, "nli_profile", None)),
        }
        self.row_list.append(row)
        self.df = pd.DataFrame.from_records(self.row_list)
        self._flushed = False

    def flush(self):
        self.df.to_csv(self.filename, quoting=csv.QUOTE_NONNUMERIC, index=False)
        self._flushed = True

    def close(self):
        # self.fout.close()
        super().close()

    def __del__(self):
        if not self._flushed:
            logger.warning("CSVLogger exiting without calling flush().")
