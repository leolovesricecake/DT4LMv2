"""

Determine successful in untargeted Classification
----------------------------------------------------
"""

from .classification_goal_function import ClassificationGoalFunction


class DifferentialClassification(ClassificationGoalFunction):
    """ A differential attack on classification models which attempts to trigger differential behaviors.
    It minimizes the score of the correct label for model1 and maximizes the score of correct label for model2
    until model1 predict wrongly while model2 predict correctly.
    """

    def __init__(self, *args, **kwargs):
        # model_1_wrapper, model_2_wrapper = args
        # print("model 1: ", model_1_wrapper.__dict__)
        # print("model 2: ", model_2_wrapper.__dict__)
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_1_output, model_2_output, _): # model_2_output (find where _is_goal_complete called, pass in model_2_output)
        if (model_1_output.numel() == 1) and isinstance(self.ground_truth_output, float): # this indicates that it is a regression task, for a classification task, model's output will have more than 1 number (the logits for different classes) + the label is int instead of float
            raise ValueError("Sorry, the differential classification goal function currently only supports classification tasks.")
        else:
            # return True if model1 predicts wrongly and model2 predicts correctly; else False
            return ((model_1_output.argmax() != self.ground_truth_output) and (model_2_output.argmax() == self.ground_truth_output))

    def _get_score(self, model_1_output, model_2_output, args, _):
        # If the model outputs a single number and the ground truth output is
        # a float, we assume that this is a regression task.
        if (model_1_output.numel() == 1) and isinstance(self.ground_truth_output, float):
            raise ValueError("Sorry, the differential classification goal function currently only supports classification tasks.")
        else:
            lambda1 = 1 + (model_1_output[self.ground_truth_output] - 0.5)
            lambda2 = 1 + (0.5 - model_2_output[self.ground_truth_output])
            if ((model_1_output.argmax() == self.ground_truth_output) and (model_2_output.argmax() == self.ground_truth_output)):
                return model_2_output[self.ground_truth_output] - lambda1*model_1_output[self.ground_truth_output] + 0.5
            elif ((model_1_output.argmax() == self.ground_truth_output) and (model_2_output.argmax() != self.ground_truth_output)):
                return model_2_output[self.ground_truth_output] - model_1_output[self.ground_truth_output]
            elif ((model_1_output.argmax() != self.ground_truth_output) and (model_2_output.argmax() != self.ground_truth_output)):
                return lambda2*model_2_output[self.ground_truth_output] - model_1_output[self.ground_truth_output] + 0.5
            else: # already achieved goals
                return pow(10,6)