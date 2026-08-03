"""Pure tests for DT4LM improvements that do not require torch downloads."""

from dataclasses import replace
from enum import Enum
import csv
import importlib.util
import json
from collections import OrderedDict
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _namespace(name):
    """Create a lightweight namespace without importing TextAttack's root."""

    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return sys.modules[name]


def _load(name, relative_path):
    """Load one repository module under its real import name."""

    parts = name.split(".")
    for index in range(1, len(parts)):
        _namespace(".".join(parts[:index]))
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


classification = _load(
    "textattack.models.classification_output",
    "textattack/models/classification_output.py",
)
objectives = _load(
    "textattack.goal_functions.classification.differential_objectives",
    "textattack/goal_functions/classification/differential_objectives.py",
)
classification_goal_stub = types.ModuleType(
    "textattack.goal_functions.classification.classification_goal_function"
)
classification_goal_stub.ClassificationGoalFunction = type(
    "ClassificationGoalFunction", (), {}
)
sys.modules[
    "textattack.goal_functions.classification.classification_goal_function"
] = classification_goal_stub
differential_goal = _load(
    "textattack.goal_functions.classification.differential_classification",
    "textattack/goal_functions/classification/differential_classification.py",
)
schemas = _load(
    "textattack.semantic_validation.schemas",
    "textattack/semantic_validation/schemas.py",
)
collection = _load(
    "textattack.semantic_validation.candidate_collection",
    "textattack/semantic_validation/candidate_collection.py",
)
thresholds = _load(
    "textattack.semantic_validation.threshold_search",
    "textattack/semantic_validation/threshold_search.py",
)
audit = _load(
    "textattack.semantic_validation.distribution_audit",
    "textattack/semantic_validation/distribution_audit.py",
)
judge_base = _load(
    "textattack.semantic_validation.judges.base",
    "textattack/semantic_validation/judges/base.py",
)
openai_judge = _load(
    "textattack.semantic_validation.judges.openai_responses",
    "textattack/semantic_validation/judges/openai_responses.py",
)
human_sample = _load(
    "dt4lm_statistics.sample_human_evaluation",
    "statistics/sample_human_evaluation.py",
)
human_analysis = _load(
    "dt4lm_statistics.analyze_human_evaluation",
    "statistics/analyze_human_evaluation.py",
)
evaluation = _load(
    "dt4lm_statistics.evaluate_improvements",
    "statistics/evaluate_improvements.py",
)
aggregation = _load(
    "dt4lm_statistics.aggregate_improvements",
    "statistics/aggregate_improvements.py",
)
metric_recomputation = _load(
    "dt4lm_statistics.recompute_metrics",
    "statistics/recompute_metrics.py",
)
artifacts = _load(
    "dt4lm_artifacts",
    "dt4lm_artifacts.py",
)

# BidirectionalNLI's aggregation can be tested without tensor inference. The
# lightweight stubs let the module load while each score is supplied directly.
torch_stub = types.ModuleType("torch")
# SciPy's array compatibility probe performs isinstance(value, torch.Tensor).
torch_stub.Tensor = type("Tensor", (), {})
sys.modules.setdefault("torch", torch_stub)


class Constraint:
    def __init__(self, compare_against_original):
        self.compare_against_original = compare_against_original


constraints_module = _namespace("textattack.constraints")
constraints_module.Constraint = Constraint
shared_module = _namespace("textattack.shared")
shared_module.logger = types.SimpleNamespace(
    info=lambda *args, **kwargs: None,
    warning=lambda *args, **kwargs: None,
)
logger_base_module = types.ModuleType("textattack.loggers.logger")
logger_base_module.Logger = type("Logger", (), {"close": lambda self: None})
sys.modules["textattack.loggers.logger"] = logger_base_module
jsonl_logger = _load(
    "textattack.loggers.jsonl_logger",
    "textattack/loggers/jsonl_logger.py",
)
nli_module = _load(
    "textattack.constraints.semantics.bidirectional_nli",
    "textattack/constraints/semantics/bidirectional_nli.py",
)
hf_judge_module = _load(
    "textattack.semantic_validation.judges.huggingface_causal",
    "textattack/semantic_validation/judges/huggingface_causal.py",
)

# Metric and dataset base classes are irrelevant to these pure calculations.
metric_module = types.ModuleType("textattack.metrics.metric")
metric_module.Metric = object
sys.modules["textattack.metrics.metric"] = metric_module
metrics = _load(
    "textattack.metrics.attack_metrics.differential_metrics",
    "textattack/metrics/attack_metrics/differential_metrics.py",
)
dataset_module = types.ModuleType("textattack.datasets.dataset")
dataset_module.Dataset = object
sys.modules["textattack.datasets.dataset"] = dataset_module
package_sampling = _load(
    "textattack.dt4lm_sampling",
    "textattack/dt4lm_sampling.py",
)
manifests = _load(
    "textattack.datasets.manifest",
    "textattack/datasets/manifest.py",
)
dataset_protocol = _load(
    "dt4lm_dataset",
    "dt4lm_dataset.py",
)
manifest_preparation = _load(
    "dt4lm_statistics.prepare_manifests",
    "statistics/prepare_manifests.py",
)
experiment_runner = _load(
    "dt4lm_experiments.run_improvements",
    "experiments/improvements/run_improvements.py",
)
semdt_calibrator = _load(
    "dt4lm_experiments.calibrate_semdt",
    "experiments/improvements/calibrate_semdt.py",
)
data_preprocessing = _load(
    "dt4lm_data.preprocess_dataset",
    "datasets/preprocess_dataset.py",
)


class Status(Enum):
    SEARCHING = 1
    SUCCEEDED = 2


status_module = _namespace("textattack.goal_function_results")
status_module.GoalFunctionResultStatus = Status
comparators = _load(
    "textattack.search_methods.differential_comparators",
    "textattack/search_methods/differential_comparators.py",
)
search_method_module = types.ModuleType("textattack.search_methods.search_method")
search_method_module.SearchMethod = object
sys.modules["textattack.search_methods.search_method"] = search_method_module
greedy_search = _load(
    "textattack.search_methods.comparator_greedy_search",
    "textattack/search_methods/comparator_greedy_search.py",
)


def output(values, score_type="probabilities"):
    return classification.ClassificationModelOutput(
        np.asarray(values, dtype=float), score_type
    )


def candidate(identifier, label=0, entailment=0.5, contradiction=0.5):
    """Create compact scored records for split and weighting tests."""

    return schemas.CandidateRecord(
        candidate_id=str(identifier),
        dataset="sst2",
        split="train",
        dataset_index=int(identifier),
        ground_truth_label=label,
        label_name=str(label),
        original_fields={"text": "original"},
        candidate_fields={"text": f"candidate {identifier}"},
        changed_fields=["text"],
        modified_indices=[0],
        modification_cost=0.1,
        search_round=0,
        candidate_order=int(identifier),
        model_pair_query=int(identifier) + 1,
        entailment_score=entailment,
        contradiction_score=contradiction,
    )


def test_explicit_logits_are_never_guessed_as_probabilities():
    result = output([0.2, 0.8], "logits")
    assert result.score_type == "logits"
    assert result.logits.tolist() == [0.2, 0.8]
    assert not np.allclose(result.probabilities, result.scores)


def test_probability_fallback_uses_log_margin():
    result = output([0.25, 0.75])
    expected = np.log(0.75 + 1e-12) - np.log(0.25 + 1e-12)
    assert result.margin(1, label_should_win=True) == pytest.approx(expected)
    assert result.margin(1, label_should_win=False) == pytest.approx(-expected)


def test_invalid_probability_contract_fails():
    with pytest.raises(ValueError, match="sum to 1"):
        output([0.2, 0.7])
    with pytest.raises(ValueError, match="explicitly declare"):
        classification.require_wrapper_score_type(object())


def test_dynamic_static_and_success_match_definitions():
    new = output([0.7, 0.3])
    old = output([0.8, 0.2])
    dynamic = objectives.DynamicObjective().score(new, old, 0, 0.2)
    assert dynamic == pytest.approx(0.8 - 1.2 * 0.7 + 0.5)
    assert objectives.StaticObjective().score(new, old, 0, 0.2) == pytest.approx(
        0.1
    )
    assert not objectives.differential_success(new, old, 0)
    assert objectives.differential_success(
        output([0.4, 0.6]), old, 0
    )


def test_differential_goal_skips_only_preexisting_differentials():
    goal = differential_goal.DifferentialClassification.__new__(
        differential_goal.DifferentialClassification
    )
    goal.ground_truth_output = 0
    correct = output([0.8, 0.2])
    wrong = output([0.2, 0.8])
    assert not goal._should_skip_2(correct, correct, None)
    assert not goal._should_skip_2(correct, wrong, None)
    assert not goal._should_skip_2(wrong, wrong, None)
    assert goal._should_skip_2(wrong, correct, None)


def test_jsonl_status_helpers_keep_failed_and_skipped_distinct():
    original = types.SimpleNamespace(
        ground_truth_output=0,
        new_model_output={"predicted_label": 1},
        old_model_output={"predicted_label": 0},
    )
    assert jsonl_logger._initial_state(original) == "already_differential"
    successful = type("SuccessfulAttackResult", (), {})()
    failed = type("FailedAttackResult", (), {})()
    skipped = type("SkippedAttackResult", (), {})()
    assert jsonl_logger._result_status(successful) == "successful"
    assert jsonl_logger._result_status(failed) == "failed"
    assert jsonl_logger._result_status(skipped) == "skipped"


def test_jsonl_preserves_mrpc_and_mr_structured_fields(tmp_path):
    """Serialize pair and single-text tasks without printable-prefix parsing."""

    class Text:
        def __init__(self, fields, dataset_index):
            self.text_input = OrderedDict(fields)
            self.attack_attrs = {
                "dataset_index": dataset_index,
                "run_config_hash": "hash",
                "modified_indices": {0},
            }

        def modification_rate(self, _):
            return 0.25

    for index, (original_fields, candidate_fields) in enumerate(
        (
            (
                (("sentence1", "a"), ("sentence2", "b")),
                (("sentence1", "a2"), ("sentence2", "b")),
            ),
            ((("text", "bad film"),), (("text", "poor film"),)),
        )
    ):
        original = types.SimpleNamespace(
            attacked_text=Text(original_fields, index),
            ground_truth_output=1,
            num_queries=1,
            new_model_output={"predicted_label": 1},
            old_model_output={"predicted_label": 1},
        )
        perturbed = types.SimpleNamespace(
            attacked_text=Text(candidate_fields, index),
            score=1.0,
            objective_name="dynamic",
            new_model_output={"predicted_label": 0},
            old_model_output={"predicted_label": 1},
        )
        result = type("SuccessfulAttackResult", (), {})()
        result.original_result = original
        result.perturbed_result = perturbed
        result.num_queries = 5
        result.wall_clock_seconds = 0.1
        result.peak_vram_bytes = 0
        result.nli_profile = None
        logger = jsonl_logger.JSONLLogger(str(tmp_path / f"{index}.jsonl"))
        logger.log_attack_result(result)
        assert logger._rows[0]["original_input"] == dict(original_fields)
        assert logger._rows[0]["candidate_input"] == dict(candidate_fields)
        logger.flush()


def test_lexi_uses_prediction_state_for_tied_argmax():
    # NumPy argmax chooses label 0 on a tie, so label 1 is explicitly wrong
    # even though the corresponding margin is exactly zero.
    tied_new = output([0.5, 0.5], "logits")
    tied_old = output([0.5, 0.5], "logits")
    score = objectives.LexicographicObjective().score(
        tied_new, tied_old, label=1, modification_cost=0.25
    )
    assert score.old_is_correct == 0
    assert score.new_is_incorrect == 1
    assert score.old_margin == 0
    assert score.new_margin == 0
    assert score.negative_cost == -0.25


def test_scalar_and_lexi_comparators_prioritize_success_correctly():
    class Result:
        def __init__(self, score, status):
            self.score = score
            self.goal_status = status

    scalar_results = [
        Result(10.0, Status.SEARCHING),
        Result(1.0, Status.SUCCEEDED),
        Result(1.0, Status.SUCCEEDED),
    ]
    assert comparators.ScalarComparator().select(scalar_results) is scalar_results[1]
    lexi_results = [
        Result(objectives.LexiScore(1, 0, 1, 0, -0.2), Status.SUCCEEDED),
        Result(objectives.LexiScore(1, 0, 1, 0, -0.1), Status.SUCCEEDED),
    ]
    assert comparators.LexicographicComparator().select(lexi_results) is lexi_results[1]


def test_shared_greedy_search_keeps_stable_order_and_metadata():
    class Text:
        def __init__(self, name):
            self.name = name
            self.attack_attrs = {}

    class Result:
        def __init__(self, text, score, status=Status.SEARCHING):
            self.attacked_text = text
            self.score = score
            self.goal_status = status

    initial = Result(Text("initial"), 0.0)
    first = Text("first")
    second = Text("second")
    search = greedy_search.ComparatorGreedySearch(
        comparators.ScalarComparator()
    )
    calls = []

    def transformations(current, original_text):
        calls.append((current.name, original_text.name))
        return [first, second] if current is initial.attacked_text else []

    search.get_transformations = transformations
    search.get_goal_results = lambda texts: (
        [Result(texts[0], 1.0), Result(texts[1], 1.0)],
        False,
    )
    result = search.perform_search(initial)
    assert result.attacked_text is first
    assert first.attack_attrs == {"search_round": 0, "candidate_order": 0}
    assert second.attack_attrs == {"search_round": 0, "candidate_order": 1}
    assert calls == [("initial", "initial"), ("first", "initial")]


def test_shared_greedy_search_propagates_unexpected_errors():
    initial = types.SimpleNamespace(
        attacked_text=types.SimpleNamespace(attack_attrs={}),
        goal_status=Status.SEARCHING,
    )
    search = greedy_search.ComparatorGreedySearch()

    def fail(*_, **__):
        raise RuntimeError("transformation failed")

    search.get_transformations = fail
    with pytest.raises(RuntimeError, match="transformation failed"):
        search.perform_search(initial)


def test_manifest_selection_is_stable_and_dataset_agnostic():
    selected = manifests.select_sample_indices(
        20,
        sample_size=7,
        seed=765,
        dataset_fingerprint="fingerprint",
        split="test",
    )
    assert len(selected) == 7
    assert selected == manifests.select_sample_indices(
        20,
        sample_size=7,
        seed=765,
        dataset_fingerprint="fingerprint",
        split="test",
    )
    for size in (None, 0, -1, 20, 100):
        assert manifests.select_sample_indices(
            20,
            sample_size=size,
            seed=765,
            dataset_fingerprint="fingerprint",
            split="test",
        ) == list(range(20))


def test_package_sampling_adapter_exports_selection_helpers():
    """Keep dataset imports working when only the textattack package is mapped."""

    assert package_sampling.select_sample_indices(
        20,
        sample_size=7,
        seed=765,
        dataset_fingerprint="fingerprint",
        split="test",
    ) == manifests.select_sample_indices(
        20,
        sample_size=7,
        seed=765,
        dataset_fingerprint="fingerprint",
        split="test",
    )


def test_attacker_does_not_bind_sample_manifest_to_a_model_pair():
    """Prevent stale model fields from leaking back into dataset manifests."""

    source = (ROOT / "textattack/attacker.py").read_text(encoding="utf-8")
    for field in (
        "model_pair_id",
        "new_model_id",
        "new_model_revision",
        "old_model_id",
        "old_model_revision",
    ):
        assert f"manifest.{field}" not in source
    assert "manifest.dataset_revision or manifest.dataset_fingerprint" in source
    assert '"manifest_selection_sha256": manifest.selection_sha256' in source


def test_manifest_view_rejects_a_changed_dataset_fingerprint():
    manifest = manifests.SampleManifest(
        schema_version=2,
        dataset_id="sst2",
        dataset_fingerprint="frozen-fingerprint",
        dataset_revision=None,
        split="test",
        population_size=2,
        requested_sample_size=None,
        effective_sample_size=2,
        seed=765,
        sampling_algorithm="all_v1",
        selected_indices=[0, 1],
        selection_sha256=(
            manifests.selection_hash(
                "sst2", "frozen-fingerprint", "test", 765, [0, 1]
            )
        ),
    )

    class SourceDataset:
        """Minimal dataset carrying the Hugging Face fingerprint contract."""

        _dataset = types.SimpleNamespace(_fingerprint="changed-fingerprint")
        input_columns = ("sentence",)
        label_names = ["negative", "positive"]
        label_map = None
        output_scale_factor = None

        def __len__(self):
            return 2

    with pytest.raises(ValueError, match="fingerprint"):
        manifests.ManifestDatasetView(SourceDataset(), manifest)


def test_manifest_preparation_uses_configured_hyperparameter():
    class Split:
        _fingerprint = "fingerprint"

        def __len__(self):
            return 10

    dataset = Split()
    config = {"id": "sst2", "revision": None}
    sampling = {
        "split": "test",
        "sample_size": 4,
        "sample_seed": 765,
    }
    manifest = manifest_preparation.build_manifest(config, dataset, sampling)
    assert manifest["requested_sample_size"] == 4
    assert manifest["effective_sample_size"] == 4


def test_experiment_configs_define_independent_runtime_axes():
    config_root = ROOT / "experiments/improvements/configs"
    system_recipes = {
        "dt4lm-kuleshov": "kuleshov_var",
        "dt4lm-leap": "leap",
        "dt4lm-fastga": "faster-alzantot",
    }
    controlled = {
        "dynamic-beam": ("dynamic", 5, None),
        "ff-pareto-greedy": ("feasibility_pareto", 1, "fill"),
        "hard-pbs": ("feasibility_pareto", 5, "discard"),
        "ff-mnew": ("feasibility_mnew", 5, "fill"),
        "ff-pbs": ("feasibility_pareto", 5, "fill"),
    }
    for dataset_id in ("sst2", "rte", "mrpc", "mr"):
        for model_pair in ("albertbasev1-v2", "gpt1-2"):
            for name, recipe in system_recipes.items():
                config = experiment_runner.load_experiment_config(
                    config_root / dataset_id / f"{model_pair}-{name}.yaml"
                )
                assert config["attack"]["recipe"] == recipe
                assert config["attack"]["search"] == {"method": "recipe_native"}
                assert config["attack"]["recipe_parameters"]
            for name, axes in controlled.items():
                config = experiment_runner.load_experiment_config(
                    config_root / dataset_id / f"{model_pair}-{name}.yaml"
                )
                assert config["attack"]["differential_objective"] == "dynamic"
                assert config["attack"]["semantic_constraint"] == "original"
                assert config["semantic"]["threshold"]["source"] == "none"
                search = config["attack"]["search"]
                assert (
                    search["ranking"],
                    search["beam_size"],
                    search.get("infeasible_state_policy"),
                ) == axes


def test_dataset_configs_expose_sampling_and_task_schemas():
    config_dir = ROOT / "experiments/improvements/configs"
    configs = {
        dataset_id: experiment_runner.load_experiment_config(
            config_dir / dataset_id / "albertbasev1-v2-dt4lm-kuleshov.yaml"
        )
        for dataset_id in ("sst2", "rte", "mrpc", "mr")
    }
    sst2 = configs["sst2"]
    rte = configs["rte"]
    assert sst2["dataset"]["evaluation"]["sample_size"] == 1000
    assert rte["dataset"]["evaluation"]["sample_size"] == 1000
    assert sst2["dataset"]["path"] == "outputs/datasets/sst2"
    assert sst2["dataset"]["evaluation"]["split"] == "test"
    assert rte["dataset"]["path"] == "outputs/datasets/rte"
    assert rte["dataset"]["text_columns"] == ["premise", "hypothesis"]
    assert configs["mrpc"]["dataset"]["text_columns"] == [
        "sentence1",
        "sentence2",
    ]
    assert configs["mr"]["dataset"]["text_columns"] == ["text"]
    assert "paraphrases" in configs["mrpc"]["dataset"]["task_definition"]
    assert "sentiment" in configs["mr"]["dataset"]["task_definition"]
    for config in configs.values():
        assert config["models"]["id"] == "albertbasev1-v2"
        namespace = Path(config["dataset"]["evaluation"]["manifest"]).parent.name
        assert namespace == config["dataset"]["id"]
        assert config["calibration"] == {"enabled": False}
        assert config["evaluation"]["core"]["success_budgets"] == list(
            range(100, 1001, 100)
        )


def test_dataset_schema_preflight_supports_mrpc_and_mr():
    """Validate both pair and single-text schemas before any model is loaded."""

    class Split:
        def __init__(self, columns):
            self.column_names = columns

    mrpc = {
        "id": "mrpc",
        "text_columns": ["sentence1", "sentence2"],
        "label_column": "label",
    }
    mr = {
        "id": "mr",
        "text_columns": ["text"],
        "label_column": "label",
    }
    assert dataset_protocol.validate_dataset_split_schema(
        mrpc, Split(["sentence1", "sentence2", "label"]), "test"
    ) == ("sentence1", "sentence2", "label")
    assert dataset_protocol.validate_dataset_split_schema(
        mr, Split(["text", "label"]), "test"
    ) == ("text", "label")
    with pytest.raises(ValueError, match="schema mismatch"):
        dataset_protocol.validate_dataset_split_schema(
            mrpc, Split(["premise", "hypothesis", "label"]), "test"
        )


def test_notebook_preprocessing_cli_has_reproducible_local_defaults():
    assert set(data_preprocessing.DATASET_SPECS) == {
        "sst2",
        "rte",
        "mrpc",
        "mr",
    }
    rte = data_preprocessing.DATASET_SPECS["rte"]
    assert rte.source == "nyu-mll/glue"
    assert rte.subset == "rte"
    assert rte.rename_columns == (
        ("sentence1", "premise"),
        ("sentence2", "hypothesis"),
    )
    mrpc = data_preprocessing.DATASET_SPECS["mrpc"]
    assert mrpc.source == "nyu-mll/glue"
    assert mrpc.subset == "mrpc"
    assert mrpc.rename_columns == ()
    mr = data_preprocessing.DATASET_SPECS["mr"]
    assert mr.source == "cornell-movie-review-data/rotten_tomatoes"
    assert mr.subset is None
    assert mr.shuffle_complete_splits is True
    parsed = data_preprocessing.build_parser().parse_args(["sst2"])
    assert parsed.output_dir == "outputs/datasets/<dataset>"
    assert parsed.seed == 42
    assert data_preprocessing.default_output_dir("sst2") == (
        ROOT / "outputs/datasets/sst2"
    )


def test_rte_notebook_conversion_filters_and_stratifies(monkeypatch):
    import pandas as pd
    import random

    class Feature:
        """Minimal stand-in for datasets.ClassLabel."""

        names = ["entailment", "not_entailment"]

    class FakeSplit:
        """Exercise preprocessing behavior without downloading HF data."""

        def __init__(self, rows, label_feature=None):
            self.rows = [dict(row) for row in rows]
            self.features = {"label": label_feature or Feature()}

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, key):
            if isinstance(key, str):
                return [row[key] for row in self.rows]
            return self.rows[key]

        @property
        def column_names(self):
            return list(self.rows[0]) if self.rows else []

        def filter(self, predicate):
            return FakeSplit(
                [row for row in self.rows if predicate(row)],
                self.features["label"],
            )

        def to_pandas(self):
            return pd.DataFrame(self.rows)

        @classmethod
        def from_pandas(cls, frame, preserve_index=False):
            assert preserve_index is False
            return cls(frame.to_dict(orient="records"))

        def cast_column(self, name, feature):
            result = FakeSplit(self.rows, feature)
            return result

        def remove_columns(self, columns):
            columns = set(columns)
            return FakeSplit(
                [
                    {key: value for key, value in row.items() if key not in columns}
                    for row in self.rows
                ],
                self.features["label"],
            )

        def rename_column(self, source, target):
            return FakeSplit(
                [
                    {
                        (target if key == source else key): value
                        for key, value in row.items()
                    }
                    for row in self.rows
                ],
                self.features["label"],
            )

        def shuffle(self, seed):
            rows = list(self.rows)
            random.Random(seed).shuffle(rows)
            return FakeSplit(rows, self.features["label"])

    class FakeDatasetDict(dict):
        pass

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = FakeSplit
    fake_datasets.DatasetDict = FakeDatasetDict
    fake_datasets.concatenate_datasets = lambda splits: FakeSplit(
        [row for split in splits for row in split.rows]
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    def rows(start, count, label_offset=0):
        return [
            {
                "sentence1": f"premise-{index}",
                "sentence2": f"hypothesis-{index}",
                "label": (
                    -1 if label_offset == -1 else (index + label_offset) % 2
                ),
                "idx": index,
            }
            for index in range(start, start + count)
        ]

    source = FakeDatasetDict(
        {
            "train": FakeSplit(rows(0, 40)),
            "validation": FakeSplit(rows(40, 20)),
            "test": FakeSplit(rows(60, 10, label_offset=-1)),
        }
    )
    processed = data_preprocessing.preprocess_dataset(
        source,
        data_preprocessing.DATASET_SPECS["rte"],
        seed=42,
    )
    assert {name: len(split) for name, split in processed.items()} == {
        "train": 48,
        "validation": 6,
        "test": 6,
    }
    for split in processed.values():
        assert split.column_names == ["premise", "hypothesis", "label"]
        assert set(split["label"]) == {0, 1}


def test_attack_command_uses_each_experiment_file_instead_of_a_matrix(tmp_path):
    config_dir = ROOT / "experiments/improvements/configs/sst2"
    for path in config_dir.glob("*.yaml"):
        config = experiment_runner.load_experiment_config(path)
        threshold_config = config["semantic"]["threshold"]
        if threshold_config["source"] == "calibrated":
            threshold = tmp_path / path.stem / "threshold.json"
            threshold.parent.mkdir(parents=True, exist_ok=True)
            threshold.write_text(
                json.dumps(
                    {
                        "judge_backend": threshold_config["backend"],
                        "dataset": config["dataset"]["id"],
                        "model_pair_id": config["models"]["id"],
                    }
                ),
                encoding="utf-8",
            )
            threshold_config["artifact"] = str(threshold)
        command = experiment_runner._attack_command(
            config,
            tmp_path / "run",
            tmp_path / "manifest.json",
            ROOT,
        )
        assert command[command.index("--differential-objective") + 1] == (
            config["attack"]["differential_objective"]
        )
        assert command[command.index("--semantic-constraint") + 1] == (
            config["attack"]["semantic_constraint"]
        )
        source = threshold_config["source"]
        assert ("--semantic-threshold-file" in command) == (
            source == "calibrated"
        )
        assert ("--nli-entailment-threshold" in command) == (source == "manual")
        assert "--do-not-push" in command


def test_experiment_attacks_do_not_enter_legacy_schema_parser():
    """Keep structured runs independent of printable dataset field prefixes."""

    source = (ROOT / "textattack/attacker.py").read_text(encoding="utf-8")
    guarded_export = "not self.attack_args.do_not_push\n                and isinstance"
    assert source.count(guarded_export) == 2
    assert source.count("no matching data schema for the current task") == 2


def test_calibration_rejects_reusing_a_different_frozen_split(tmp_path):
    split = tmp_path / "split_manifest.json"
    split.write_text(
        json.dumps(
            {
                "seed": 765,
                "search_ids": ["a", "b"],
                "validation_ids": ["c"],
            }
        ),
        encoding="utf-8",
    )
    calibration = {"candidate_sample_size": 3, "search_sample_size": 2}
    semdt_calibrator._verify_frozen_split(split, calibration, 765)
    with pytest.raises(ValueError, match="configured seed"):
        semdt_calibrator._verify_frozen_split(split, calibration, 123)


def test_calibration_rejects_candidates_from_another_model_pair(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "dataset": "sst2",
                "metadata": {
                    "model_pair_id": "pair-a",
                    "old_model_id": "old",
                    "new_model_id": "new",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "dataset": {"id": "sst2"},
        "models": {"id": "pair-a", "old": "old", "new": "new"},
    }
    semdt_calibrator._verify_candidate_scope(candidates, config)
    config["models"]["id"] = "pair-b"
    with pytest.raises(ValueError, match="model pairs"):
        semdt_calibrator._verify_candidate_scope(candidates, config)


def test_calibration_rejects_threshold_from_another_judge(tmp_path):
    threshold = tmp_path / "threshold.json"
    threshold.write_text(
        json.dumps(
            {
                "judge_backend": "openai",
                "judge_model": "model-a",
                "threshold_search_method": "grid",
                "threshold_step": 0.01,
                "min_precision": 0.95,
                "dataset": "sst2",
                "model_pair_id": "pair-a",
            }
        ),
        encoding="utf-8",
    )
    search = {"method": "grid", "step": 0.01, "min_precision": 0.95}
    semdt_calibrator._verify_threshold_identity(
        threshold, "openai", "model-a", search, "sst2", "pair-a"
    )
    with pytest.raises(ValueError, match="expected"):
        semdt_calibrator._verify_threshold_identity(
            threshold, "hf", "model-b", search, "sst2", "pair-b"
        )


def test_model_pair_namespace_and_run_directory_are_mandatory(tmp_path):
    config = {
        "experiment": {
            "id": "base",
            "output_root": str(tmp_path / "outputs"),
        },
        "dataset": {
            "id": "sst2",
            "evaluation": {
                "manifest": str(tmp_path / "sample_sets" / "sst2" / "test.json")
            },
        },
        "models": {
            "id": "old-new",
            "old": {"name_or_path": "old"},
            "new": {"name_or_path": "new"},
        },
        "calibration": {"enabled": False},
    }
    artifacts.validate_artifact_namespaces(config, ROOT)
    assert artifacts.run_directory(config, ROOT, "base") == (
        tmp_path / "outputs" / "runs" / "sst2" / "old-new" / "base"
    )

    config["models"]["id"] = "unsafe/pair"
    with pytest.raises(ValueError, match="models.id"):
        artifacts.validate_artifact_namespaces(config, ROOT)


def test_manifest_identity_rejects_another_dataset_before_attack(tmp_path):
    config = {
        "dataset": {"id": "sst2"},
        "models": {
            "id": "pair-a",
            "old": {"name_or_path": "old"},
            "new": {"name_or_path": "new"},
        },
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset_id": "rte",
                "dataset_fingerprint": "fingerprint",
                "dataset_revision": None,
                "split": "test",
                "population_size": 1,
                "requested_sample_size": None,
                "effective_sample_size": 1,
                "seed": 765,
                "sampling_algorithm": "all_v1",
                "selected_indices": [0],
                "selection_sha256": manifests.selection_hash(
                    "rte", "fingerprint", "test", 765, [0]
                ),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dataset split"):
        artifacts.validate_manifest_identity(manifest, config, "test", ROOT)


def test_manifest_artifacts_are_immutable_after_creation(tmp_path):
    first = tmp_path / "train_manifest.json"
    second = tmp_path / "test_manifest.json"
    manifest_preparation._write_frozen_payloads(
        [(first, {"dataset_id": "sst2"}), (second, {"split": "test"})]
    )
    # Reusing byte-semantically identical artifacts is an idempotent operation.
    manifest_preparation._write_frozen_payloads(
        [(first, {"dataset_id": "sst2"}), (second, {"split": "test"})]
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        manifest_preparation._write_frozen_payloads(
            [(first, {"dataset_id": "rte"}), (second, {"split": "test"})]
        )
    assert json.loads(first.read_text(encoding="utf-8"))["dataset_id"] == "sst2"


def test_stratified_1000_and_frozen_800_200_are_deterministic():
    rows = [
        candidate(
            index,
            label=index % 2,
            entailment=(index % 10 + 0.1) / 10,
            contradiction=((index // 10) % 10 + 0.1) / 10,
        )
        for index in range(1200)
    ]
    sampled = collection.stratified_candidate_sample(rows, seed=765)
    search, validation = collection.freeze_calibration_split(sampled, seed=765)
    assert len(sampled) == 1000
    assert len(search) == 800
    assert len(validation) == 200
    assert {row.candidate_id for row in search}.isdisjoint(
        row.candidate_id for row in validation
    )
    assert sampled == collection.stratified_candidate_sample(rows, seed=765)
    assert all(row.inclusion_weight >= 1 for row in sampled)


def test_supplemental_sample_excludes_initial_candidates():
    rows = [candidate(index) for index in range(2100)]
    supplemental = collection.supplemental_audit_sample(
        rows,
        excluded_candidate_ids=(str(index) for index in range(1000)),
        maximum_size=1000,
        seed=765,
    )
    assert len(supplemental) == 1000
    assert not ({row.candidate_id for row in supplemental} & {str(i) for i in range(1000)})


def test_threshold_search_obeys_precision_floor_and_frozen_report():
    examples = [
        thresholds.WeightedSemanticExample("a", 0.9, 0.1, True),
        thresholds.WeightedSemanticExample("b", 0.8, 0.2, True),
        thresholds.WeightedSemanticExample("c", 0.7, 0.3, False),
    ]
    entailment, contradiction, result = thresholds.search_thresholds(
        examples, min_precision=1.0, step=0.1
    )
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert entailment <= 0.8
    assert contradiction >= 0.2
    report = thresholds.validation_report(
        examples, entailment, contradiction, bootstrap_samples=20
    )
    assert report["metrics"]["precision"] == 1.0


def test_threshold_search_fails_without_feasible_acceptance():
    examples = [
        thresholds.WeightedSemanticExample("negative", 1.0, 0.0, False)
    ]
    with pytest.raises(ValueError, match="No threshold pair"):
        thresholds.search_thresholds(examples, min_precision=0.95)


def test_qps_uses_every_manifest_sample_query():
    records = [
        {"result_status": "successful", "model_pair_queries": 10, "modification_rate": 0.2},
        {"result_status": "failed", "model_pair_queries": 1000, "modification_rate": 0},
        {"result_status": "successful", "model_pair_queries": 20, "modification_rate": 0.4},
    ]
    result = metrics.calculate_differential_metrics(records, sample_count=3)
    assert result["model_pair_qps"] == 515
    assert result["paper_gsr"] == pytest.approx(2 / 3)
    assert result["successful"] == 2
    assert result["failed"] == 1
    assert result["skipped"] == 0
    assert result["amr"] == pytest.approx(0.3)


def test_zero_success_qps_is_json_null_value():
    result = metrics.calculate_differential_metrics(
        [{"result_status": "skipped", "model_pair_queries": 9, "modification_rate": 0}],
        sample_count=1,
    )
    assert result["model_pair_qps"] is None
    assert result["model_pair_query_total"] == 9


def test_trajectory_sampling_is_weighted_and_cannot_retune():
    records = [
        {
            "candidate_id": str(index),
            "entailment_score": (index % 5 + 0.1) / 5,
            "contradiction_score": ((index // 5) % 5 + 0.1) / 5,
            "accepted": index % 2 == 0,
        }
        for index in range(300)
    ]
    sampled = audit.sample_trajectory_audit(records, sample_size=100)
    assert len(sampled) == 100
    assert all("inclusion_weight" in row for row in sampled)
    report = audit.audit_fixed_threshold(
        sampled,
        {row["candidate_id"]: True for row in sampled},
        entailment_threshold=0.5,
        contradiction_threshold=0.5,
    )
    assert report["threshold_changed"] is False
    shift = audit.distribution_shift_report(
        sampled[:50],
        sampled[50:],
        entailment_threshold=0.5,
        contradiction_threshold=0.5,
    )
    assert shift["threshold_changed"] is False
    assert 0 <= shift["weighted_ks_entailment"] <= 1
    assert 0 <= shift["weighted_ks_contradiction"] <= 1


def test_nli_label_lookup_is_name_based():
    entailment, contradiction = nli_module._normalized_label_indices(
        {7: "CONTRADICTION", 3: "neutral", 1: "Entailment"}
    )
    assert entailment == 1
    assert contradiction == 7
    with pytest.raises(ValueError, match="must name"):
        nli_module._normalized_label_indices({0: "yes", 1: "no"})


def test_nli_changed_field_aggregation_boundaries_and_cache_counts():
    class Text:
        def __init__(self, fields):
            self.text_input = fields
            self.attack_attrs = {}

    reference = Text(OrderedDict([("premise", "p"), ("hypothesis", "h")]))
    candidate_one = Text(
        OrderedDict([("premise", "p"), ("hypothesis", "h1")])
    )
    candidate_two = Text(
        OrderedDict([("premise", "p2"), ("hypothesis", "h2")])
    )
    scores = {
        ("h", "h1"): nli_module.NLIFieldScore(0.9, 0.95, 0.05, 0.04),
        ("p", "p2"): nli_module.NLIFieldScore(0.92, 0.91, 0.03, 0.04),
        ("h", "h2"): nli_module.NLIFieldScore(
            0.90, 0.93, 0.02, 0.05, forward_truncated=True
        ),
    }
    nli = nli_module.BidirectionalNLI.__new__(nli_module.BidirectionalNLI)
    nli.model_name_or_path = "fake"
    nli.model_revision = "model-rev"
    nli.tokenizer_revision = "tokenizer-rev"
    nli.tokenizer = types.SimpleNamespace(name_or_path="fake-tokenizer")
    nli.max_length = 128
    nli.truncation_strategy = "longest_first"
    nli.entailment_threshold = 0.90
    nli.contradiction_threshold = 0.05
    nli.audit_log = None
    nli.cache_size = 100
    nli._cache = OrderedDict()
    nli.profile = nli_module.NLIProfile()

    def score_pending(self, pending):
        for key, pair in pending.items():
            self._cache_put(key, scores[pair])

    nli._score_pending_fields = types.MethodType(score_pending, nli)
    accepted = nli._check_constraint_many(
        [candidate_one, candidate_two], reference
    )
    # Exact 0.90/0.05 values pass because both threshold comparisons are closed.
    assert accepted == [candidate_one, candidate_two]
    assert candidate_two.attack_attrs["nli"]["entailment_score"] == 0.90
    assert candidate_two.attack_attrs["nli"]["contradiction_score"] == 0.05
    assert candidate_two.attack_attrs["nli"]["changed_fields"] == [
        "premise",
        "hypothesis",
    ]
    assert nli.profile.cache_misses == 3
    assert nli.profile.cache_hits == 0
    assert nli.profile.logical_directional_pairs == 6
    assert nli.profile.truncated_candidates == 1

    nli._check_constraint_many([candidate_one, candidate_two], reference)
    assert nli.profile.cache_misses == 3
    assert nli.profile.cache_hits == 3
    old_key = nli._cache_key("h", "h1")
    nli.max_length = 256
    assert nli._cache_key("h", "h1") != old_key


def test_secret_pattern_and_example_are_safe():
    gitignore = (ROOT.parent / ".gitignore").read_text(encoding="utf-8")
    example = (ROOT / "configs/semantic_judge.example.yaml").read_text(
        encoding="utf-8"
    )
    assert "**.secert.yaml" in gitignore
    assert "**.secret.yaml" in gitignore
    assert "replace-me" in example
    assert "deepseek-v4-pro" in example


def test_judge_schema_is_strict_and_config_repr_redacts_key():
    assert judge_base.parse_boolean_response(
        '{"semantic_preserved": true}'
    ) is True
    with pytest.raises(ValueError, match="strict boolean"):
        judge_base.parse_boolean_response(
            '{"semantic_preserved": true, "reason": "extra"}'
        )
    config = judge_base.JudgeConfig(
        backend="openai", model="mock", api_key="sk-private"
    )
    assert "sk-private" not in repr(config)
    assert "***" in repr(config)


def test_openai_responses_judge_uses_structured_output_and_redacts_errors():
    class Responses:
        def __init__(self):
            self.arguments = None

        def create(self, **kwargs):
            self.arguments = kwargs
            return {
                "id": "request-1",
                "status": "completed",
                "output_text": '{"semantic_preserved": true}',
                "usage": {"input_tokens": 7, "output_tokens": 4},
            }

    class Client:
        def __init__(self):
            self.responses = Responses()

    config = judge_base.JudgeConfig(
        backend="openai",
        model="mock-model",
        base_url="https://example.invalid",
        api_key="sk-private",
        max_retries=0,
    )
    client = Client()
    judge = openai_judge.OpenAIResponsesJudge(config, client=client)
    example = schemas.JudgeExample(
        candidate_id="candidate",
        dataset="sst2",
        task_definition="sentiment",
        ground_truth_label=1,
        label_name="positive",
        original_fields={"text": "good"},
        candidate_fields={"text": "great"},
    )
    result = judge.annotate([example])[0]
    assert result.semantic_preserved is True
    assert result.prompt_text
    assert client.responses.arguments["temperature"] == 0
    assert (
        client.responses.arguments["text"]["format"]["type"] == "json_schema"
    )

    class FailingResponses:
        def create(self, **_):
            raise RuntimeError("credential sk-private rejected")

    failing_client = types.SimpleNamespace(responses=FailingResponses())
    failure = openai_judge.OpenAIResponsesJudge(
        config, client=failing_client
    ).annotate([example])[0]
    assert not failure.success
    assert "sk-private" not in failure.error_message
    assert "***" in failure.error_message


def test_hf_judge_repairs_only_invalid_json_format():
    judge = hf_judge_module.HuggingFaceCausalLMJudge.__new__(
        hf_judge_module.HuggingFaceCausalLMJudge
    )
    judge.config = judge_base.JudgeConfig(
        backend="hf", model="mock-hf", max_retries=1
    )
    judge._metadata = lambda: {"revision": "mock"}
    repairs = []

    def repair(prompt):
        repairs.append(prompt)
        return '{"semantic_preserved": false}'

    judge._repair_once = repair
    example = schemas.JudgeExample(
        candidate_id="candidate",
        dataset="rte",
        task_definition="entailment",
        ground_truth_label=0,
        label_name="entailment",
        original_fields={"premise": "p", "hypothesis": "h"},
        candidate_fields={"premise": "p2", "hypothesis": "h"},
    )
    result = judge._parse_with_repairs(
        example, "shared prompt", "not-json", started_at=0.0
    )
    assert result.success
    assert result.semantic_preserved is False
    assert result.attempts == 2
    assert repairs == ["shared prompt"]


def test_human_sampling_and_joint_validity_rates():
    def row(index, status):
        return {
            "schema_version": 4,
            "dataset_index": index,
            "result_status": status,
            "original_input": {"sentence": f"original-{index}"},
            "candidate_input": {"sentence": f"candidate-{index}"},
            "ground_truth_output": 0,
        }

    kuleshov = {
        0: row(0, "successful"),
        1: row(1, "successful"),
        2: row(2, "failed"),
    }
    ff = {0: row(0, "failed"), 1: row(1, "successful"), 2: row(2, "successful")}
    reviews, key = human_sample.build_sample(
        kuleshov, ff, method_sample_size=2, unique_sample_size=1, seed=7
    )
    assert len(reviews) == 4
    assert key["population_counts"] == {
        "kuleshov_overall": 2,
        "ffpbs_overall": 2,
        "ffpbs_unique": 1,
    }
    unique = [item for item in key["rows"] if "ffpbs_unique" in item["cohorts"]]
    assert len(unique) == 1
    assert unique[0]["method"] == "FF-PBS"

    observations = [
        {"label_preserved": True, "semantic_preserved": True},
        {"label_preserved": True, "semantic_preserved": False},
        {"label_preserved": False, "semantic_preserved": True},
    ]
    rates = human_analysis._rates(observations)
    assert rates["lpr"] == pytest.approx(2 / 3)
    assert rates["spr"] == pytest.approx(2 / 3)
    assert rates["hvr"] == pytest.approx(1 / 3)


def test_core_metrics_preserve_curve_data_without_cross_run_comparison():
    manifest = {
        "effective_sample_size": 3,
        "selected_indices": [3, 5, 8],
    }
    records = [
        {
            "schema_version": 4,
            "dataset_index": 3,
            "result_status": "successful",
            "initial_state": "both_correct",
            "model_pair_queries": 10,
            "queries_to_success": 10,
            "modification_rate": 0.2,
        },
        {
            "schema_version": 4,
            "dataset_index": 5,
            "result_status": "failed",
            "initial_state": "both_wrong",
            "model_pair_queries": 1000,
            "queries_to_success": None,
            "modification_rate": 0.0,
        },
        {
            "schema_version": 4,
            "dataset_index": 8,
            "result_status": "skipped",
            "initial_state": "already_differential",
            "model_pair_queries": 1,
            "queries_to_success": None,
            "modification_rate": 0.0,
        },
    ]
    result, query_data = evaluation.core_metrics(
        records,
        manifest,
        success_budgets=[100, 500, 1000],
        query_budget=1000,
    )
    assert (result["successful"], result["failed"], result["skipped"]) == (1, 1, 1)
    assert result["paper_gsr"] == 0.5
    assert result["sample_generation_rate"] == pytest.approx(1 / 3)
    assert result["model_pair_qps"] == 1011
    assert "success_query_data" not in result
    assert "successful_query_counts" not in result
    assert query_data["data"] == {
        "dataset_index": [3, 5, 8],
        "result_status": ["successful", "failed", "skipped"],
        "model_pair_queries": [10, 1000, 1],
        "queries_to_success": [10, None, None],
        "budget_penalized_queries": [10, 1000, None],
    }
    assert "comparison_to_base" not in result


def test_missing_local_bertscore_model_is_an_isolated_quality_failure(tmp_path):
    records = [
        {
            "result_status": "successful",
            "original_input": {"sentence": "a good film"},
            "candidate_input": {"sentence": "a fine film"},
        }
    ]
    quality = {
        "bleu": {"enabled": False},
        "meteor": {"enabled": False},
        "rouge_l": {"enabled": False},
        "bertscore": {
            "enabled": True,
            "model_name_or_path": str(tmp_path / "missing-model"),
            "num_layers": 17,
            "allow_remote_download": False,
            "device": "cpu",
            "batch_size": 2,
            "idf": False,
            "rescale_with_baseline": False,
            "baseline_path": None,
        },
    }
    summary = evaluation.run_quality_metrics(
        records, quality, tmp_path / "metrics", ROOT
    )
    assert summary["status"] == "failed"
    assert summary["metrics"]["bertscore"]["status"] == "failed"
    assert not (tmp_path / "metrics/bertscore.json").exists()
    persisted = json.loads(
        (tmp_path / "metrics/quality.json").read_text(encoding="utf-8")
    )
    assert set(persisted["metrics"]) == {
        "bleu",
        "meteor",
        "rouge_l",
        "bertscore",
    }


def test_bertscore_preflight_requires_secure_pickle_loading(tmp_path):
    model = tmp_path / "deberta"
    model.mkdir()
    (model / "pytorch_model.bin").write_bytes(b"not-loaded-by-this-test")
    with pytest.raises(RuntimeError, match="CVE-2025-32434"):
        evaluation._local_model_weight_format(model, "2.4.1+cu118")
    assert evaluation._local_model_weight_format(model, "2.6.0+cu118") == "pytorch_bin"

    (model / "model.safetensors").write_bytes(b"not-loaded-by-this-test")
    assert evaluation._local_model_weight_format(model, "2.4.1+cu118") == "safetensors"


def test_quality_retry_reuses_completed_metrics_and_retries_failure(tmp_path):
    records = [
        {
            "result_status": "successful",
            "original_input": {"sentence": "original"},
            "candidate_input": {"sentence": "candidate"},
        }
    ]
    quality = {
        "bleu": {"enabled": True},
        "meteor": {"enabled": True},
        "rouge_l": {"enabled": True},
        "bertscore": {
            "enabled": True,
            "model_name_or_path": str(tmp_path / "missing-model"),
            "num_layers": 17,
            "allow_remote_download": False,
            "device": "cpu",
            "batch_size": 2,
            "idf": False,
            "rescale_with_baseline": False,
            "baseline_path": None,
        },
    }
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    completed = {
        name: {
            "status": "completed",
            "config": quality[name],
            "values": {"value": index, "sample_count": 1},
        }
        for index, name in enumerate(("bleu", "meteor", "rouge_l"), start=1)
    }
    prior = {
        "schema_version": 4,
        "status": "failed",
        "successful_sample_count": 1,
        "metrics": {
            **completed,
            "bertscore": {
                "status": "failed",
                "config": quality["bertscore"],
                "values": None,
            },
        },
    }
    (metrics_dir / "quality.json").write_text(json.dumps(prior), encoding="utf-8")
    retried = evaluation.run_quality_metrics(records, quality, metrics_dir, ROOT)
    assert retried["metrics"]["bleu"]["values"]["value"] == 1
    assert retried["metrics"]["meteor"]["values"]["value"] == 2
    assert retried["metrics"]["rouge_l"]["values"]["value"] == 3
    assert retried["metrics"]["bertscore"]["status"] == "failed"


def test_runner_recognizes_complete_schema_v4_metric_checkpoints(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "core.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "total": 2,
                "successful": 1,
                "resources": {},
                "query_budget": 1000,
                "success_at_100": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (metrics / "query_data.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "sample_count": 2,
                "successful_sample_count": 1,
                "data": {
                    "dataset_index": [4, 5],
                    "result_status": ["successful", "failed"],
                    "model_pair_queries": [9, 1000],
                    "queries_to_success": [9, None],
                    "budget_penalized_queries": [9, 1000],
                },
            }
        ),
        encoding="utf-8",
    )
    quality_config = {
        "bleu": {"enabled": True},
        "meteor": {"enabled": False},
        "rouge_l": {"enabled": False},
        "bertscore": {"enabled": False},
    }
    (metrics / "quality.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "status": "completed",
                "successful_sample_count": 1,
                "metrics": {
                    name: {
                        "status": "completed" if config["enabled"] else "disabled",
                        "config": config,
                    }
                    for name, config in quality_config.items()
                },
            }
        ),
        encoding="utf-8",
    )
    assert experiment_runner._core_evaluation_complete(
        tmp_path, {"success_budgets": [100]}, 1000
    )
    assert experiment_runner._quality_evaluation_complete(tmp_path, quality_config)


def test_aggregate_improvements_writes_paper_metrics(tmp_path):
    input_dir = tmp_path / "run"
    run_dir = input_dir / "sst2" / "pair" / "experiment"
    metrics = run_dir / "metrics"
    metrics.mkdir(parents=True)
    config = {
        "experiment": {"id": "experiment", "method": "base", "seed": 765},
        "dataset": {"id": "sst2"},
        "models": {
            "id": "pair",
            "old": {"name_or_path": "old", "revision": None},
            "new": {"name_or_path": "new", "revision": None},
        },
        "attack": {
            "recipe": "kuleshov_var",
            "differential_objective": "dynamic",
            "semantic_constraint": "original",
            "query_budget": 1000,
        },
        "semantic": {"threshold": {"source": "none"}},
        "calibration": {"enabled": False},
    }
    (run_dir / "config.resolved.yaml").write_text(
        __import__("yaml").safe_dump(config), encoding="utf-8"
    )
    core = {
        "schema_version": 4,
        "total": 2,
        "attackable": 2,
        "successful": 1,
        "failed": 1,
        "skipped": 0,
        "query_budget": 1000,
        "successful_query_count": 1,
        "initial_state_counts": {
            "both_correct": 2,
            "new_correct_old_wrong": 0,
            "both_wrong": 0,
            "already_differential": 0,
        },
        "paper_gsr": 0.5,
        "sample_generation_rate": 0.5,
        "success_at_100": 0.5,
        "model_pair_qps": 12,
        "resources": {"end_to_end_seconds": 3.5, "peak_vram_bytes": 10},
    }
    quality = {
        "schema_version": 4,
        "status": "completed",
        "successful_sample_count": 1,
        "metrics": {
            "bleu": {
                "status": "completed",
                "config": {"enabled": True},
                "values": {"value": 0.8, "sample_count": 1},
            },
            "meteor": {"status": "disabled", "config": {"enabled": False}},
            "rouge_l": {"status": "disabled", "config": {"enabled": False}},
            "bertscore": {"status": "disabled", "config": {"enabled": False}},
        },
    }
    query_data = {
        "schema_version": 4,
        "sample_count": 2,
        "attackable_sample_count": 2,
        "successful_sample_count": 1,
        "query_budget": 1000,
        "data": {
            "dataset_index": [7, 8],
            "result_status": ["successful", "failed"],
            "model_pair_queries": [12, 1000],
            "queries_to_success": [12, None],
            "budget_penalized_queries": [12, 1000],
        },
    }
    manifest = {
        "effective_sample_size": 2,
        "population_size": 100,
        "seed": 765,
        "split": "test",
        "selection_sha256": "abc",
        "selected_indices": [7, 8],
    }
    status = {
        "attack": {"status": "completed"},
        "core_evaluation": {"status": "completed"},
    }
    provenance = {
        "git_commit": "deadbeef",
        "packages": {"torch": "2.6.0", "transformers": "4.57.6"},
        "gpus": ["GPU"],
    }
    for path, payload in (
        (metrics / "core.json", core),
        (metrics / "quality.json", quality),
        (metrics / "query_data.json", query_data),
        (run_dir / "sample_manifest.json", manifest),
        (run_dir / "status.json", status),
        (run_dir / "provenance.json", provenance),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")

    output = input_dir / "paper.csv"
    assert aggregation.write_summary(input_dir, output) == 1
    with open(output, encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert (row["dataset"], row["model_pair"], row["method"], row["seed"]) == (
        "sst2",
        "pair",
        "base",
        "765",
    )
    assert row["paper_gsr"] == "0.5"
    assert row["bleu"] == "0.8"
    assert row["metrics_schema_version"] == "4"


def test_metric_recomputation_discovers_nested_runs(tmp_path):
    first = tmp_path / "rte" / "pair" / "first"
    second = tmp_path / "sst2" / "pair" / "second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "config.resolved.yaml").write_text("experiment: {}", encoding="utf-8")
    (second / "config.resolved.yaml").write_text("experiment: {}", encoding="utf-8")
    assert metric_recomputation.discover_runs(tmp_path) == [first, second]


def test_manual_metric_retry_updates_only_its_stage_status(tmp_path):
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps({"attack": {"status": "completed"}}), encoding="utf-8"
    )
    evaluation._update_stage_status(
        status_path, "quality_evaluation", "completed"
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["attack"]["status"] == "completed"
    assert status["quality_evaluation"]["status"] == "completed"


def test_attack_summary_persists_all_three_result_counts(tmp_path):
    summary_path = tmp_path / "attack_summary.json"
    summary_path.write_text(json.dumps({"Attack Results": {}}), encoding="utf-8")
    records = [
        {"result_status": "successful"},
        {"result_status": "failed"},
        {"result_status": "skipped"},
        {"result_status": "failed"},
    ]
    experiment_runner._augment_attack_summary(summary_path, records)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["result_counts"] == {
        "total": 4,
        "successful": 1,
        "failed": 2,
        "skipped": 1,
    }
