

class TreeNode:
    def __init__(self, score, samples_per_class, predicted_class):
        self.score = score
        self.samples_per_class = samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = None
        self.threshold = None
        self.left_node = None
        self.right_node = None
        self.tree_structure = {}

    def __str__(self):
        return f'Feature {self.feature_index} with score of {self.score} and split of {self.threshold}'

    def __repr__(self):
        return self.__str__()
