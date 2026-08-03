"""PAIR recipe for DT4LM objectives and bounded-frontier experiments."""

import textattack
from textattack.goal_functions import DifferentialClassification
from textattack.model_args import ModelArgs
from textattack.search_methods import (
    AsyncDifferentialBeamSearch,
    ComparatorGreedySearch,
)
from textattack.search_methods.differential_comparators import (
    comparator_for_objective,
)

from .a2t_yoo_2021 import A2TYoo2021
from .bae_garg_2019 import BAEGarg2019
from .bert_attack_li_2020 import BERTAttackLi2020
from .checklist_ribeiro_2020 import CheckList2020
from .clare_li_2020 import CLARE2020
from .deepwordbug_gao_2018 import DeepWordBugGao2018
from .faster_genetic_algorithm_jia_2019 import FasterGeneticAlgorithmJia2019
from .genetic_algorithm_alzantot_2018 import GeneticAlgorithmAlzantot2018
from .hotflip_ebrahimi_2017 import HotFlipEbrahimi2017
from .iga_wang_2019 import IGAWang2019
from .input_reduction_feng_2018 import InputReductionFeng2018
from .kuleshov_2017 import Kuleshov2017
from .kuleshov_2017_var import Kuleshov2017Var
from .leap_2023 import LEAP2023
from .morpheus_tan_2020 import MorpheusTan2020
from .pruthi_2019 import Pruthi2019
from .pso_zang_2020 import PSOZang2020
from .pwws_ren_2019 import PWWSRen2019
from .seq2sick_cheng_2018_blackbox import Seq2SickCheng2018BlackBox
from .textbugger_li_2018 import TextBuggerLi2018
from .textfooler_jin_2019 import TextFoolerJin2019

from .attack_recipe import AttackRecipe


BASE_RECIPES = {
    "alzantot": GeneticAlgorithmAlzantot2018,
    "bae": BAEGarg2019,
    "bert-attack": BERTAttackLi2020,
    "faster-alzantot": FasterGeneticAlgorithmJia2019,
    "deepwordbug": DeepWordBugGao2018,
    "hotflip": HotFlipEbrahimi2017,
    "input-reduction": InputReductionFeng2018,
    "kuleshov": Kuleshov2017,
    "morpheus": MorpheusTan2020,
    "seq2sick": Seq2SickCheng2018BlackBox,
    "textbugger": TextBuggerLi2018,
    "textfooler": TextFoolerJin2019,
    "pwws": PWWSRen2019,
    "iga": IGAWang2019,
    "pruthi": Pruthi2019,
    "pso": PSOZang2020,
    "checklist": CheckList2020,
    "clare": CLARE2020,
    "a2t": A2TYoo2021,
    "leap": LEAP2023,
    "kuleshov_var": Kuleshov2017Var,
}


class PAIR2024(AttackRecipe):
    """Build a model-pair attack from orthogonal objective/semantic switches."""

    @staticmethod
    def build(model_wrapper, args):
        try:
            base_recipe = BASE_RECIPES[args.base_recipe]
        except KeyError as exc:
            raise ValueError(f"Unknown PAIR base recipe {args.base_recipe!r}.") from exc

        old_model_wrapper = ModelArgs._create_second_model_from_args(args)
        ModelArgs.validate_classification_model_pair(
            model_wrapper, old_model_wrapper
        )
        recipe_parameters = args.base_recipe_parameters or {}
        if not isinstance(recipe_parameters, dict):
            raise ValueError("--base-recipe-parameters must decode to a JSON object.")
        try:
            base_attack = base_recipe.build(model_wrapper, **recipe_parameters)
        except TypeError as exc:
            raise ValueError(
                f"Invalid parameters for base recipe {args.base_recipe!r}: {exc}"
            ) from exc

        goal_function = DifferentialClassification(
            model_wrapper,
            old_model_wrapper,
            attack_args=args,
            objective=args.differential_objective,
        )
        constraints = list(
            base_attack.pre_transformation_constraints + base_attack.constraints
        )

        if args.semantic_constraint == "nli":
            # Importing lazily keeps non-NLI recipes free of the NLI model
            # dependency and avoids allocating its memory for unrelated runs.
            from textattack.constraints.semantics import BidirectionalNLI

            constraints.append(BidirectionalNLI.from_attack_args(args))

        if args.candidate_log:
            from textattack.semantic_validation.candidate_collection import (
                CandidateObserver,
            )

            # Persist all provenance available at recipe construction time.
            # Dataset fingerprints and source indices are added by the manifest
            # and attacker once concrete samples have been loaded.
            metadata = {
                "dataset": args.experiment_dataset_id
                or args.dataset_from_huggingface
                or args.dataset_by_model
                or args.dataset_from_file
                or "",
                "model_pair_id": args.model_pair_id,
                "split": args.dataset_split or "train",
                "new_model_id": args.model
                or args.model_from_huggingface
                or args.model_from_file,
                "new_model_revision": args.model_revision,
                "old_model_id": args.second_model,
                "old_model_revision": args.second_model_revision,
                "recipe": args.base_recipe,
                "seed": args.random_seed,
                "query_budget": args.query_budget,
            }
            goal_function.set_candidate_observer(
                CandidateObserver(args.candidate_log, metadata=metadata)
            )

        if args.base_recipe != "kuleshov_var":
            if (
                args.differential_objective != "dynamic"
                or args.semantic_constraint != "original"
                or args.differential_search
                not in {"recipe_native", "legacy_greedy"}
            ):
                raise ValueError(
                    "Non-Kuleshov recipes support only the dynamic objective, "
                    "original recipe constraints, and recipe-native search."
                )
            # Existing LEAP/FastGA/etc. pair commands retain their own search
            # state machine. Only the differential goal is rebound, matching
            # the legacy PAIR recipe behavior.
            base_attack.goal_function = goal_function
            base_attack.search_method.goal_function = goal_function
            base_attack.search_method.get_goal_results = goal_function.get_results
            return base_attack

        if args.differential_search in {"recipe_native", "legacy_greedy"}:
            search_method = ComparatorGreedySearch(
                comparator_for_objective(args.differential_objective)
            )
        else:
            if args.differential_objective != "dynamic":
                raise ValueError(
                    "Asynchronous differential search requires "
                    "--differential-objective dynamic in its first version."
                )
            search_method = AsyncDifferentialBeamSearch(
                ranking=args.differential_frontier_ranking,
                beam_size=args.differential_beam_size,
                infeasible_state_policy=args.infeasible_state_policy,
                trace_output=args.search_trace_output,
            )
        return textattack.Attack(
            goal_function,
            constraints,
            base_attack.transformation,
            search_method,
        )
