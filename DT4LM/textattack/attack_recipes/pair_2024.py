"""

PAIR
==================================

(PAPER-NAME)

"""
import textattack
import transformers
from textattack import CommandLineAttackArgs
from textattack.goal_functions import DifferentialClassification

from .attack_recipe import AttackRecipe

from .a2t_yoo_2021 import A2TYoo2021
from .bae_garg_2019 import BAEGarg2019
from .bert_attack_li_2020 import BERTAttackLi2020
from .genetic_algorithm_alzantot_2018 import GeneticAlgorithmAlzantot2018
from .faster_genetic_algorithm_jia_2019 import FasterGeneticAlgorithmJia2019
from .deepwordbug_gao_2018 import DeepWordBugGao2018
from .hotflip_ebrahimi_2017 import HotFlipEbrahimi2017
from .input_reduction_feng_2018 import InputReductionFeng2018
from .kuleshov_2017 import Kuleshov2017
from .morpheus_tan_2020 import MorpheusTan2020
from .seq2sick_cheng_2018_blackbox import Seq2SickCheng2018BlackBox
from .textbugger_li_2018 import TextBuggerLi2018
from .textfooler_jin_2019 import TextFoolerJin2019
from .pwws_ren_2019 import PWWSRen2019
from .iga_wang_2019 import IGAWang2019
from .pruthi_2019 import Pruthi2019
from .pso_zang_2020 import PSOZang2020
from .checklist_ribeiro_2020 import CheckList2020
from .clare_li_2020 import CLARE2020
from .leap_2023 import LEAP2023
from .kuleshov_2017_var import Kuleshov2017Var


class PAIR2024(AttackRecipe):
    @staticmethod
    def build(model_wrapper, args):
        # map the base recipe to its class
        map_to_class = {"alzantot": GeneticAlgorithmAlzantot2018, "bae": BAEGarg2019, "bert-attack": BERTAttackLi2020, "faster-alzantot": FasterGeneticAlgorithmJia2019, "deepwordbug": DeepWordBugGao2018, "hotflip": HotFlipEbrahimi2017, "input-reduction": InputReductionFeng2018, "kuleshov": Kuleshov2017, "morpheus": MorpheusTan2020, "seq2sick": Seq2SickCheng2018BlackBox, "textbugger": TextBuggerLi2018, "textfooler": TextFoolerJin2019, "pwws": PWWSRen2019, "iga": IGAWang2019, "pruthi": Pruthi2019, "pso": PSOZang2020, "checklist": CheckList2020, "clare": CLARE2020, "a2t": A2TYoo2021, "leap": LEAP2023, "kuleshov_var": Kuleshov2017Var}
        # when empty, only reading in default values
        base_recipe = args.base_recipe

        # construct the attack based on the base recipe, with modified goal function
        base_attack = map_to_class[base_recipe].build(model_wrapper)
        # during this step, the original goal function will be initialized, and hence the log files might show another goal function name, which is fine, the next line will show DifferentialClassification, which means that the goal function has been reinitialized.
        
        if not args.second_model:
            raise Exception("For model pair attack, it is a must to specify the second model using the --model2 argument")
        
        # load second model to conduct model pair attack
        model_2_name = args.second_model
        model_2 = transformers.AutoModelForSequenceClassification.from_pretrained(model_2_name)
        model_2_tokenizer = transformers.AutoTokenizer.from_pretrained(model_2_name, use_fast=True)
        model_2_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model_2, model_2_tokenizer)
        pair_goal_function = DifferentialClassification(model_wrapper, model_2_wrapper, attack_args=args)

        base_attack.goal_function = pair_goal_function
        base_attack.search_method.goal_function = pair_goal_function
        base_attack.search_method.get_goal_results = pair_goal_function.get_results

        return base_attack