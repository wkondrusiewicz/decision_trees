from collections import Counter

import numpy as np


class BaseScorer:

    def __init__(self):
        pass

    @staticmethod
    def get_score(y):
        pass

    def get_score_after_split(self, X, y, threshold):
        pass

    @staticmethod
    def scoring_condition(previous_value, new_value):
        pass


class GiniImportance(BaseScorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def scoring_condition(previous_value, new_value):
        return previous_value < new_value

    @staticmethod
    def get_score(y):
        frequencies = np.array([*Counter(y).values()]) / len(y)
        gini = 1 - np.sum(frequencies**2)
        return gini

    def get_score_after_split(self, X, y, threshold):
        left_mask = X < threshold
        right_mask = X >= threshold
        y_left = y[left_mask]
        y_right = y[right_mask]
        left_gini = self.get_score(y_left)
        right_gini = self.get_score(y_right)
        return (len(y_left) * left_gini + len(y_right) * right_gini) / len(y)


class MeanSquaredError(BaseScorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def scoring_condition(previous_value, new_value):
        return previous_value < new_value

    @staticmethod
    def get_score(y):
        return np.sum(np.array(y)**2)
        return gini

    def get_score_after_split(self, X, y, threshold):
        left_mask = X < threshold
        right_mask = X >= threshold
        y_left = y[left_mask]
        y_right = y[right_mask]
        left_mse = self.get_score(y_left)
        right_mse = self.get_score(y_right)
        return (len(y_left) * left_mse + len(y_right) * right_mse) / len(y)
