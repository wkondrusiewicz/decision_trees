

class TreeNode:
    def __init__(self, score):
        self.score = score
        self.feature_index = None
        self.threshold = None
        self.left_node = None
        self.right_node = None

    def __str__(self):
        return f'Feature {self.feature_index} with score of {self.score} and split of {self.threshold}'

    def __repr__(self):
        return self.__str__()
