"""Configuration and runner regression tests for the FF-PBS matrix."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from improvement_config import load_experiment_config, validate_experiment_config
from experiments.improvements.run_improvements import _attack_command


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "experiments" / "improvements" / "configs"
METHODS = {
    "dt4lm-kuleshov",
    "dt4lm-leap",
    "dt4lm-fastga",
    "dynamic-beam",
    "ff-pareto-greedy",
    "hard-pbs",
    "ff-mnew",
    "ff-pbs",
    "ff-pbs-k3",
    "ff-pbs-k10",
    "ffms-greedy",
    "hard-ffms",
}


class ConfigTests(unittest.TestCase):
    def test_wordnet_recipe_never_downloads_during_runtime(self):
        source = (
            ROOT
            / "textattack/transformations/word_swaps/word_swap_wordnet.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("nltk.download(", source)
        self.assertIn("wordnet.ensure_loaded()", source)

    def test_active_matrix_contains_only_paper_methods(self):
        paths = sorted(CONFIG_ROOT.glob("*/*.yaml"))
        configs = [load_experiment_config(path) for path in paths]

        self.assertEqual(len(configs), 144)
        grouped = {}
        for config in configs:
            key = (config["dataset"]["id"], config["models"]["id"])
            grouped.setdefault(key, set()).add(config["experiment"]["method"])
        self.assertEqual(len(grouped), 12)
        self.assertTrue(all(methods == METHODS for methods in grouped.values()))

    def test_system_recipes_keep_native_search_and_explicit_parameters(self):
        expected = {
            "dt4lm-kuleshov": "kuleshov_var",
            "dt4lm-leap": "leap",
            "dt4lm-fastga": "faster-alzantot",
        }
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            for method, recipe in expected.items():
                config = load_experiment_config(
                    CONFIG_ROOT / "sst2" / f"albertbasev1-v2-{method}.yaml"
                )
                command = _attack_command(
                    config, temporary / method, temporary / "manifest.json", ROOT
                )
                self.assertEqual(config["attack"]["search"]["method"], "recipe_native")
                self.assertEqual(command[command.index("--base-recipe") + 1], recipe)
                parameters = json.loads(
                    command[command.index("--base-recipe-parameters") + 1]
                )
                self.assertEqual(parameters, config["attack"]["recipe_parameters"])
                self.assertEqual(
                    command[command.index("--differential-search") + 1],
                    "recipe_native",
                )
                self.assertEqual(
                    command[command.index("--model-batch-size") + 1], "32"
                )

    def test_mnew_and_hard_policies_reach_the_attack_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            manifest = temporary / "manifest.json"
            for method, ranking, policy in (
                ("ff-mnew", "feasibility_mnew", "fill"),
                ("hard-pbs", "feasibility_pareto", "discard"),
            ):
                config = load_experiment_config(
                    CONFIG_ROOT / "sst2" / f"albertbasev1-v2-{method}.yaml"
                )
                command = _attack_command(
                    config, temporary / method, manifest, ROOT
                )
                self.assertEqual(
                    command[command.index("--differential-frontier-ranking") + 1],
                    ranking,
                )
                self.assertEqual(
                    command[command.index("--infeasible-state-policy") + 1],
                    policy,
                )
                self.assertFalse(any("epsilon" in argument for argument in command))

    def test_feasibility_ranking_requires_an_explicit_policy(self):
        config = load_experiment_config(
            CONFIG_ROOT / "sst2" / "albertbasev1-v2-ff-pbs.yaml"
        )
        invalid = copy.deepcopy(config)
        del invalid["attack"]["search"]["infeasible_state_policy"]

        with self.assertRaises(ValueError):
            validate_experiment_config(invalid)

    def test_recipe_parameter_schema_rejects_missing_population_size(self):
        config = load_experiment_config(
            CONFIG_ROOT / "sst2" / "albertbasev1-v2-dt4lm-leap.yaml"
        )
        invalid = copy.deepcopy(config)
        del invalid["attack"]["recipe_parameters"]["population_size"]

        with self.assertRaises(ValueError):
            validate_experiment_config(invalid)


if __name__ == "__main__":
    unittest.main()
