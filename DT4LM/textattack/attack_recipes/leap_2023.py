"""

LEAP
==================================

(LEAP: Efficient and Automated Test Method for NLP Software)

"""
from textattack import Attack
from textattack.constraints.pre_transformation import (
    MaxModificationRate,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import LEAP
from textattack.transformations import WordSwapWordNet

from .attack_recipe import AttackRecipe


class LEAP2023(AttackRecipe):
    @staticmethod
    def build(
        model_wrapper,
        max_modification_rate=0.16,
        population_size=60,
        max_iterations=20,
        post_turn_check=True,
        max_turn_retries=20,
    ):
        """Build LEAP while preserving its published default configuration."""

        #
        # Swap words with their synonyms extracted based on the WordNet.
        #
        transformation = WordSwapWordNet()
        #
        # MaxModificationRate = 0.16 in AG's News
        #
        constraints = [
            MaxModificationRate(max_rate=max_modification_rate),
            StopwordModification(),
        ]
        #
        #
        # Use untargeted classification for demo, can be switched to targeted one
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Perform word substitution with LEAP algorithm.
        #
        search_method = LEAP(
            pop_size=population_size,
            max_iters=max_iterations,
            post_turn_check=post_turn_check,
            max_turn_retries=max_turn_retries,
        )

        return Attack(goal_function, constraints, transformation, search_method)
