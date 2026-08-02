#!/usr/bin/env python
"""Generate the complete FF-PBS experiment matrix from Base configs."""

import copy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "experiments" / "improvements" / "configs"
SUCCESS_BUDGETS = list(range(100, 1001, 100))

METHODS = {
    "dynamic-beam": {
        "method": "async_frontier",
        "ranking": "dynamic",
        "beam_size": 5,
        "diagnostics": {"trace_enabled": False},
    },
    "ff-pareto-greedy": {
        "method": "async_frontier",
        "ranking": "feasibility_pareto",
        "beam_size": 1,
        "infeasible_state_policy": "fill",
        "diagnostics": {"trace_enabled": False},
    },
    "hard-pbs": {
        "method": "async_frontier",
        "ranking": "feasibility_pareto",
        "beam_size": 5,
        "infeasible_state_policy": "discard",
        "diagnostics": {"trace_enabled": False},
    },
    "ff-mnew": {
        "method": "async_frontier",
        "ranking": "feasibility_mnew",
        "beam_size": 5,
        "infeasible_state_policy": "fill",
        "diagnostics": {"trace_enabled": False},
    },
    "ff-pbs": {
        "method": "async_frontier",
        "ranking": "feasibility_pareto",
        "beam_size": 5,
        "infeasible_state_policy": "fill",
        "diagnostics": {"trace_enabled": False},
    },
    "ff-pbs-k3": {
        "method": "async_frontier",
        "ranking": "feasibility_pareto",
        "beam_size": 3,
        "infeasible_state_policy": "fill",
        "diagnostics": {"trace_enabled": False},
    },
    "ff-pbs-k10": {
        "method": "async_frontier",
        "ranking": "feasibility_pareto",
        "beam_size": 10,
        "infeasible_state_policy": "fill",
        "diagnostics": {"trace_enabled": False},
    },
}

DEPRECATED_SUFFIXES = (
    "ae-pbs",
    "combined-openai",
    "epsilon-greedy",
    "feasibility-first-pbs",
    "lexidt",
    "semdt-hf",
    "semdt-manual",
    "semdt-openai",
    "static",
    "strict-pbs",
)


def _write_yaml(path, config):
    """Write one generated complete config with stable YAML formatting."""

    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config,
            handle,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def generate_configs():
    """Generate all methods for every active dataset/model-pair Base config."""

    written = []
    base_paths = sorted(CONFIG_ROOT.glob("*/*-base.yaml"))
    if not base_paths:
        raise ValueError(f"No Base configs found under {CONFIG_ROOT}.")

    for base_path in base_paths:
        with open(base_path, encoding="utf-8") as handle:
            base = yaml.safe_load(handle)
        dataset_id = str(base["dataset"]["id"])
        model_pair_id = str(base["models"]["id"])
        expected_name = f"{model_pair_id}-base.yaml"
        if base_path.name != expected_name:
            raise ValueError(
                f"Base config {base_path} does not match models.id={model_pair_id!r}."
            )

        # Every method reports the same dense Success@B grid used by the paper.
        base["evaluation"]["core"]["success_budgets"] = list(SUCCESS_BUDGETS)
        _write_yaml(base_path, base)
        written.append(base_path)

        for suffix in DEPRECATED_SUFFIXES:
            legacy = base_path.parent / f"{model_pair_id}-{suffix}.yaml"
            if legacy.exists():
                legacy.unlink()

        for method_name, search in METHODS.items():
            config = copy.deepcopy(base)
            config["experiment"]["id"] = (
                f"{dataset_id}-{model_pair_id}-{method_name}"
            )
            config["experiment"]["method"] = method_name
            config["attack"]["differential_objective"] = "dynamic"
            config["attack"]["semantic_constraint"] = "original"
            config["attack"]["search"] = copy.deepcopy(search)
            output = base_path.parent / f"{model_pair_id}-{method_name}.yaml"
            _write_yaml(output, config)
            written.append(output)
    return written


def main():
    for path in generate_configs():
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
