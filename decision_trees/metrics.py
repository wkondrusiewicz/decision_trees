from collections import Counter

import numpy as np


class BaseScorer:
    def __init__(self):
        pass

    @property
    def optimization_types(self):
        pass

    @staticmethod
    def get_score(y):
        pass

    def get_score_after_split(self, X, y, threshold):
        left_mask = X < threshold
        right_mask = X >= threshold
        y_left = y[left_mask]
        y_right = y[right_mask]
        return y_left, y_right

    @staticmethod
    def scoring_condition(previous_value, new_value):
        pass

    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__str__()


class GiniImportance(BaseScorer):
    def __init__(self):
        super().__init__()

    @property
    def optimization_types(self):
        return ['classification']

    @staticmethod
    def scoring_condition(previous_value, new_value):
        return previous_value < new_value

    @staticmethod
    def get_score(y):
        frequencies = np.array([*Counter(y).values()]) / len(y)
        gini = 1 - np.sum(frequencies**2)
        return gini

    def get_score_after_split(self, X, y, threshold):
        y_left, y_right = super().get_score_after_split(X, y, threshold)
        left_gini = self.get_score(y_left)
        right_gini = self.get_score(y_right)
        return (len(y_left) * left_gini + len(y_right) * right_gini) / len(y)


class StandardDeviationReduction(BaseScorer):
    def __init__(self):
        super().__init__()

    @property
    def optimization_types(self):
        return ['regression']

    @staticmethod
    def scoring_condition(previous_value, new_value):
        return previous_value < new_value

    @staticmethod
    def get_score(y):
        return np.std(y)
        return gini

    def get_score_after_split(self, X, y, threshold):
        y_left, y_right = super().get_score_after_split(X, y, threshold)
        left_std = self.get_score(y_left)
        right_std = self.get_score(y_right)
        return (len(y_left) * left_std + len(y_right) * right_std) / len(y)


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
        y_left, y_right = super().get_score_after_split(X, y, threshold)
        left_mse = self.get_score(y_left)
        right_mse = self.get_score(y_right)
        return (len(y_left) * left_mse + len(y_right) * right_mse) / len(y)
