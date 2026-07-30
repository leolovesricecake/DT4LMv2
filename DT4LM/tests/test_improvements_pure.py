"""Pure tests for DT4LM improvements that do not require torch downloads."""

from dataclasses import replace
from enum import Enum
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
shared_module.logger = types.SimpleNamespace(warning=lambda *args, **kwargs: None)
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
manifests = _load(
    "textattack.datasets.manifest",
    "textattack/datasets/manifest.py",
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


def test_manifest_selection_is_explicit_and_dataset_agnostic():
    selected = manifests.select_manifest_indices(
        range(20), strategy="random_exact", sample_size=7, seed=765
    )
    assert len(selected) == 7
    assert selected == manifests.select_manifest_indices(
        range(20), strategy="random_exact", sample_size=7, seed=765
    )
    assert manifests.select_manifest_indices(
        [4, 2, 3], strategy="all"
    ) == [2, 3, 4]
    assert len(
        manifests.select_manifest_indices(
            range(3), strategy="random_up_to", sample_size=10
        )
    ) == 3
    with pytest.raises(ValueError, match="Requested 10"):
        manifests.select_manifest_indices(
            range(3), strategy="random_exact", sample_size=10
        )
    assert manifests.jointly_correct_indices(
        [0, 1, 0], [0, 1, 1], [0, 0, 0]
    ) == [0]


def test_manifest_preparation_uses_configured_hyperparameter():
    policy = {"strategy": "random_up_to", "size": 4}
    selected = manifest_preparation._select_indices(
        range(10), policy, seed=765, role="calibration_originals"
    )
    assert len(selected) == 4
    assert selected == manifest_preparation._select_indices(
        range(10), policy, seed=765, role="calibration_originals"
    )


def test_experiment_configs_define_independent_runtime_axes():
    import yaml

    config_dir = ROOT / "experiments/improvements/configs/experiments"
    expected = {
        "base": ("dynamic", "original", "none"),
        "static": ("static", "original", "none"),
        "lexidt": ("lexi", "original", "none"),
        "semdt-manual": ("dynamic", "nli", "manual"),
        "semdt-openai": ("dynamic", "nli", "calibrated"),
        "semdt-hf": ("dynamic", "nli", "calibrated"),
        "combined": ("lexi", "nli", "calibrated"),
    }
    for name, axes in expected.items():
        experiment = yaml.safe_load(
            (config_dir / f"{name}.yaml").read_text(encoding="utf-8")
        )
        experiment_runner._validate_experiment(experiment)
        assert (
            experiment["differential_objective"],
            experiment["semantic_constraint"],
            experiment["semantic_threshold"]["source"],
        ) == axes


def test_dataset_configs_expose_sampling_and_threshold_search():
    import yaml

    config_dir = ROOT / "experiments/improvements/configs"
    sst2 = yaml.safe_load((config_dir / "sst2.yaml").read_text(encoding="utf-8"))
    rte = yaml.safe_load((config_dir / "rte.yaml").read_text(encoding="utf-8"))
    assert sst2["sampling"]["calibration_originals"]["size"] == 500
    assert sst2["sampling"]["test"] == {"strategy": "all"}
    assert rte["sampling"]["test"] == {"strategy": "all"}
    assert sst2["dataset"]["path"] == "outputs/datasets/sst2"
    assert sst2["dataset"]["test_split"] == "test"
    assert rte["dataset"]["path"] == "outputs/datasets/rte"
    assert rte["dataset"]["text_columns"] == ["premise", "hypothesis"]
    for config in (sst2, rte):
        semdt_calibrator._validate_config(config)
        assert config["calibration"]["threshold_search"] == {
            "method": "grid",
            "step": 0.01,
            "min_precision": 0.95,
            "bootstrap_samples": 10000,
        }


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
    import yaml

    config_dir = ROOT / "experiments/improvements/configs"
    config = yaml.safe_load(
        (config_dir / "sst2.yaml").read_text(encoding="utf-8")
    )
    config["calibration"]["output_root"] = str(tmp_path / "calibration")
    for backend in ("openai", "hf"):
        threshold = tmp_path / "calibration" / backend / "threshold.json"
        threshold.parent.mkdir(parents=True, exist_ok=True)
        threshold.write_text(
            json.dumps({"judge_backend": backend}),
            encoding="utf-8",
        )

    experiment_dir = config_dir / "experiments"
    for path in experiment_dir.glob("*.yaml"):
        experiment = yaml.safe_load(path.read_text(encoding="utf-8"))
        command = experiment_runner._attack_command(
            config,
            experiment,
            tmp_path / "run",
            tmp_path / "manifest.json",
            ROOT,
        )
        assert command[command.index("--differential-objective") + 1] == (
            experiment["differential_objective"]
        )
        assert command[command.index("--semantic-constraint") + 1] == (
            experiment["semantic_constraint"]
        )
        source = experiment["semantic_threshold"]["source"]
        assert ("--semantic-threshold-file" in command) == (
            source == "calibrated"
        )
        assert ("--nli-entailment-threshold" in command) == (source == "manual")


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
            }
        ),
        encoding="utf-8",
    )
    search = {"method": "grid", "step": 0.01, "min_precision": 0.95}
    semdt_calibrator._verify_threshold_identity(
        threshold, "openai", "model-a", search
    )
    with pytest.raises(ValueError, match="expected"):
        semdt_calibrator._verify_threshold_identity(
            threshold, "hf", "model-b", search
        )


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
        {"success": True, "model_pair_queries": 10, "modification_rate": 0.2},
        {"success": False, "model_pair_queries": 1000, "modification_rate": 0},
        {"success": True, "model_pair_queries": 20, "modification_rate": 0.4},
    ]
    result = metrics.calculate_differential_metrics(
        records, sample_count=3, eligible_count=3, test_split_size=10
    )
    assert result["model_pair_qps"] == 515
    assert result["perturbation_induced_gsr"] == pytest.approx(2 / 3)
    assert result["amr"] == pytest.approx(0.3)
    assert result["eligibility_rate"] == 0.3


def test_zero_success_qps_is_json_null_value():
    result = metrics.calculate_differential_metrics(
        [{"success": False, "model_pair_queries": 9, "modification_rate": 0}],
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


def test_human_sampling_minimums_and_weighted_estimators():
    groups = {
        "common_success": list(range(90)),
        "base_only_success": list(range(90, 99)),
        "semdt_only_success": [99],
    }
    allocation = human_sample._allocate(groups, 20)
    assert sum(allocation.values()) == 20
    assert allocation["common_success"] >= 5
    assert allocation["base_only_success"] >= 5
    assert allocation["semdt_only_success"] == 1

    observations = [
        {
            "stratum": "common_success",
            "labels": {"Base": True, "SemDT": True},
        },
        {
            "stratum": "common_success",
            "labels": {"Base": False, "SemDT": True},
        },
        {"stratum": "base_only_success", "labels": {"Base": True}},
        {"stratum": "semdt_only_success", "labels": {"SemDT": False}},
    ]
    estimates = human_analysis._method_estimates(
        observations,
        {
            "common_success": 20,
            "base_only_success": 10,
            "semdt_only_success": 10,
        },
        sample_count=100,
    )
    assert estimates["Base"]["semantic_preservation_rate"] == pytest.approx(
        2 / 3
    )
    assert estimates["Base"]["valid_gsr"] == pytest.approx(0.2)
    assert estimates["SemDT"]["semantic_preservation_rate"] == pytest.approx(
        2 / 3
    )
    assert estimates["SemDT"]["valid_gsr"] == pytest.approx(0.2)


def test_equivalence_report_uses_one_point_and_five_percent_rules():
    base = {
        "perturbation_induced_gsr": 0.50,
        "amr": 0.20,
        "model_pair_qps": 100,
    }
    current = {
        "perturbation_induced_gsr": 0.509,
        "amr": 0.21,
        "model_pair_qps": 105,
    }
    comparison = evaluation.compare_to_base(current, base)
    assert comparison["gsr_equivalent_within_1pp"]
    assert comparison["amr_not_worse_than_5pct"]
    assert comparison["model_pair_qps_not_worse_than_5pct"]
