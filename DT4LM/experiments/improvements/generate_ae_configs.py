#!/usr/bin/env python
"""Generate complete AE-PBS comparison configs from each dataset Base config."""

import copy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "experiments" / "improvements" / "configs"
DATASETS = ("sst2", "rte", "mrpc", "mr")
SUCCESS_BUDGETS = list(range(100, 1001, 100))

METHODS = {
    "dynamic-beam": {
        "method": "async_frontier",
        "ranking": "dynamic",
        "beam_size": 5,
        "epsilon": {"mode": "disabled"},
        "diagnostics": {"trace_enabled": False},
    },
    "strict-pbs": {
        "method": "async_frontier",
        "ranking": "epsilon_pareto",
        "beam_size": 5,
        "epsilon": {"mode": "strict"},
        "diagnostics": {"trace_enabled": False},
    },
    "epsilon-greedy": {
        "method": "async_frontier",
        "ranking": "epsilon_pareto",
        "beam_size": 1,
        "epsilon": {
            "mode": "adaptive",
            "initial_quantile": 0.75,
            "initialization_max_expansions": 2,
            "decay": "quadratic",
        },
        "diagnostics": {"trace_enabled": False},
    },
    "ae-pbs": {
        "method": "async_frontier",
        "ranking": "epsilon_pareto",
        "beam_size": 5,
        "epsilon": {
            "mode": "adaptive",
            "initial_quantile": 0.75,
            "initialization_max_expansions": 2,
            "decay": "quadratic",
        },
        "diagnostics": {"trace_enabled": False},
    },
}


def generate_configs():
    """Write every comparison as a standalone complete experiment YAML."""

    written = []
    for dataset_id in DATASETS:
        directory = CONFIG_ROOT / dataset_id
        base_path = directory / "albertbasev1-v2-base.yaml"
        with open(base_path, encoding="utf-8") as handle:
            base = yaml.safe_load(handle)
        model_pair_id = str(base["models"]["id"])
        for method_name, search in METHODS.items():
            config = copy.deepcopy(base)
            config["experiment"]["id"] = (
                f"{dataset_id}-{model_pair_id}-{method_name}"
            )
            config["experiment"]["method"] = method_name
            config["attack"]["differential_objective"] = "dynamic"
            config["attack"]["semantic_constraint"] = "original"
            config["attack"]["search"] = copy.deepcopy(search)
            config["evaluation"]["core"]["success_budgets"] = list(
                SUCCESS_BUDGETS
            )
            output = directory / f"{model_pair_id}-{method_name}.yaml"
            with open(output, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    config,
                    handle,
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                )
            written.append(output)
    return written


def main():
    for path in generate_configs():
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
