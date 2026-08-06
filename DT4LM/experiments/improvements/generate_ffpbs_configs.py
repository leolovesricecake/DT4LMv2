#!/usr/bin/env python
"""Generate the paper-facing DT4LM and FF-PBS experiment matrix."""

import copy
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from improvement_config import validate_experiment_config  # noqa: E402


CONFIG_ROOT = PROJECT_ROOT / "experiments" / "improvements" / "configs"
SUCCESS_BUDGETS = list(range(100, 1001, 100))

# These values freeze each recipe's current published/default instantiation.
# The local GPT-2 path also prevents an unrelated network lookup at run time.
KULESHOV_PARAMETERS = {
    "max_candidates": 15,
    "max_percent": 0.5,
    "thought_vector_threshold": 0.2,
    "max_log_prob_diff": 2.0,
    "fluency_model_name_or_path": "/mnt/huawei/nsq/models/openai-community/gpt2",
}
LEAP_PARAMETERS = {
    "max_modification_rate": 0.16,
    "population_size": 60,
    "max_iterations": 20,
    "post_turn_check": True,
    "max_turn_retries": 20,
}
FASTGA_PARAMETERS = {
    "max_candidates": 8,
    "max_percent": 0.2,
    "max_mse_dist": 0.5,
    "language_model_window_size": 6,
    "max_log_prob_diff": 5.0,
    # Null selects TextAttack's standard cached Learning-to-Write model. Set an
    # explicit directory in the generated YAML for an offline installation.
    "language_model_path": None,
    "population_size": 60,
    "max_iterations": 40,
    "post_crossover_check": False,
}

SYSTEM_METHODS = {
    "dt4lm-kuleshov": {
        "recipe": "kuleshov_var",
        "recipe_parameters": KULESHOV_PARAMETERS,
        "search": {"method": "recipe_native"},
    },
    "dt4lm-leap": {
        "recipe": "leap",
        "recipe_parameters": LEAP_PARAMETERS,
        "search": {"method": "recipe_native"},
    },
    "dt4lm-fastga": {
        "recipe": "faster-alzantot",
        "recipe_parameters": FASTGA_PARAMETERS,
        "search": {"method": "recipe_native"},
    },
}

CONTROLLED_METHODS = {
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
    "ffms-greedy": {
        "method": "async_frontier",
        "ranking": "feasibility_mnew",
        "beam_size": 1,
        "infeasible_state_policy": "fill",
        "diagnostics": {"trace_enabled": False},
    },
    "hard-ffms": {
        "method": "async_frontier",
        "ranking": "feasibility_mnew",
        "beam_size": 5,
        "infeasible_state_policy": "discard",
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

    # Generated matrices must fail before overwriting valid files when a
    # template accidentally collapses a model pair to one checkpoint.
    validate_experiment_config(config)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config,
            handle,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def _template_paths():
    """Find each model-pair template, including one-time legacy Base inputs."""

    templates = {}
    candidates = sorted(CONFIG_ROOT.glob("*/*-dt4lm-kuleshov.yaml"))
    candidates.extend(sorted(CONFIG_ROOT.glob("*/*-base.yaml")))
    for path in candidates:
        with open(path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        key = (str(config["dataset"]["id"]), str(config["models"]["id"]))
        previous = templates.get(key)
        if previous is not None:
            # Prefer the already migrated template during idempotent reruns.
            if path.name.endswith("-dt4lm-kuleshov.yaml"):
                templates[key] = (path, config)
            continue
        templates[key] = (path, config)
    return [templates[key] for key in sorted(templates)]


def _method_config(template, method_name, attack_overrides):
    """Create one complete config while preserving shared experiment inputs."""

    config = copy.deepcopy(template)
    dataset_id = str(config["dataset"]["id"])
    model_pair_id = str(config["models"]["id"])
    config["experiment"]["id"] = f"{dataset_id}-{model_pair_id}-{method_name}"
    config["experiment"]["method"] = method_name
    config["attack"].update(copy.deepcopy(attack_overrides))
    config["attack"]["differential_objective"] = "dynamic"
    config["attack"]["semantic_constraint"] = "original"
    config["attack"]["model_batch_size"] = 32
    config["evaluation"]["core"]["success_budgets"] = list(SUCCESS_BUDGETS)
    return config


def generate_configs():
    """Generate three DT4LM recipes and all controlled FF-PBS variants."""

    written = []
    templates = _template_paths()
    if not templates:
        raise ValueError(f"No DT4LM-Kuleshov templates found under {CONFIG_ROOT}.")

    for source_path, template in templates:
        model_pair_id = str(template["models"]["id"])
        expected_names = {
            f"{model_pair_id}-base.yaml",
            f"{model_pair_id}-dt4lm-kuleshov.yaml",
        }
        if source_path.name not in expected_names:
            raise ValueError(
                f"Template {source_path} does not match models.id={model_pair_id!r}."
            )

        for suffix in DEPRECATED_SUFFIXES:
            legacy = source_path.parent / f"{model_pair_id}-{suffix}.yaml"
            if legacy.exists():
                legacy.unlink()

        for method_name, attack_overrides in SYSTEM_METHODS.items():
            config = _method_config(template, method_name, attack_overrides)
            output = source_path.parent / f"{model_pair_id}-{method_name}.yaml"
            _write_yaml(output, config)
            written.append(output)

        controlled_attack = {
            "recipe": "kuleshov_var",
            "recipe_parameters": KULESHOV_PARAMETERS,
        }
        for method_name, search in CONTROLLED_METHODS.items():
            overrides = copy.deepcopy(controlled_attack)
            overrides["search"] = search
            config = _method_config(template, method_name, overrides)
            output = source_path.parent / f"{model_pair_id}-{method_name}.yaml"
            _write_yaml(output, config)
            written.append(output)

        legacy_base = source_path.parent / f"{model_pair_id}-base.yaml"
        if legacy_base.exists():
            legacy_base.unlink()
    return written


def main():
    for path in generate_configs():
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
