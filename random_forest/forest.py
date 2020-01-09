import numpy as np

from scipy import stats

from decision_trees.tree2 import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestBase:
    def __init__(self, X, y, n_trees, max_depth, features_max_value_type='sqrt'):
        # add random seed
        self.X = X
        self.y = y
        self.n_trees = n_trees
        if features_max_value_type == 'sqrt':
            self.n_features = int(np.sqrt(self.X.shape[1]))
        elif features_max_value_type == 'log2':
            self.n_features = int(np.log2(self.X.shape[1]))
        else:
            self.n_features = int(
                features_max_value_type(self.X.shape[1]))
        self.sampled_feature_indices = self._get_sampled_features_indices()
        self.trees = None

    def _get_sampled_features_indices(self):
        sampled_feature_indices = []
        for _ in range(self.n_trees):
            sampled_feature_indices.append(
                np.random.permutation(self.X.shape[1])[:self.n_features])
        return sampled_feature_indices


class RandomForestClassifier(RandomForestBase):
    def __init__(self, X, y, n_trees, max_depth, features_max_value_type='sqrt'):
        super().__init__(X, y, n_trees, max_depth, features_max_value_type)
        self.trees = [DecisionTreeClassifier(
            max_depth=max_depth) for _ in range(self.n_trees)]

    def fit(self):
        for feature_indices, tree in zip(self.sampled_feature_indices, self.trees):
            tree.fit(self.X[:, feature_indices], self.y)

    def predict(self):
        predictions = []
        for feature_indices, tree in zip(self.sampled_feature_indices, self.trees):
            predictions.append(tree.predict(self.X[:, feature_indices]))
        predictions = stats.mode(np.array(predictions), axis=0)
        return predictions


class RandomForestRegressor(RandomForestBase):
    def __init__(self, X, y, n_trees, max_depth, features_max_value_type='sqrt'):
        super().__init__(X, y, n_trees, max_depth, features_max_value_type)
        self.trees = [DecisionTreeRegressor(
            max_depth=max_depth) for _ in range(self.n_trees)]

    def fit(self):
        for feature_indices, tree in zip(self.sampled_feature_indices, self.trees):
            tree.fit(self.X[:, feature_indices], self.y)

    def predict(self):
        predictions = []
        for feature_indices, tree in zip(self.sampled_feature_indices, self.trees):
            predictions.append(tree.predict(self.X[:, feature_indices]))
        predictions = np.array(predictions).mean(0)
        return predictions
