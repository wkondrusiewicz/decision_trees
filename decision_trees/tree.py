


class BasicDecisionTree:
    def __init__(self, max_depth, metric=None):
        self.__max_depth = max_depth
        self.metric = metric

    @property
    def max_depth(self):
        return self.__max_depth

    @max_depth.setter
    def max_depth(self, max_depth):
        self.__max_depth = max_depth

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def get_best_split(self, X, y):
        pass

    def grow_tree(self, X, y):
        pass

class DecisionTreeClassifier(BasicDecisionTree):
    def __init__(self, max_depth, metric):
        super().__init__(max_depth, metric)
        
