"""Configuration and runner regression tests for the FF-PBS matrix."""

import copy
from pathlib import Path
import tempfile
import unittest

from improvement_config import load_experiment_config, validate_experiment_config
from experiments.improvements.run_improvements import _attack_command


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "experiments" / "improvements" / "configs"
METHODS = {
    "base",
    "dynamic-beam",
    "ff-pareto-greedy",
    "hard-pbs",
    "ff-mnew",
    "ff-pbs",
    "ff-pbs-k3",
    "ff-pbs-k10",
}


class ConfigTests(unittest.TestCase):
    def test_active_matrix_contains_only_paper_methods(self):
        paths = sorted(CONFIG_ROOT.glob("*/*.yaml"))
        configs = [load_experiment_config(path) for path in paths]

        self.assertEqual(len(configs), 64)
        grouped = {}
        for config in configs:
            key = (config["dataset"]["id"], config["models"]["id"])
            grouped.setdefault(key, set()).add(config["experiment"]["method"])
        self.assertEqual(len(grouped), 8)
        self.assertTrue(all(methods == METHODS for methods in grouped.values()))

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


if __name__ == "__main__":
    unittest.main()
