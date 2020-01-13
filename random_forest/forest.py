import numpy as np

from scipy import stats

from decision_trees.tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestBase:
    def __init__(self, X, y, n_trees, max_depth, features_max_value_type='sqrt', rows_percentage=0.8, random_seed=42):
        self.X = X
        self.y = y
        self.n_trees = n_trees
        self.rows_percentage = rows_percentage
        if features_max_value_type == 'sqrt':
            self.n_features = int(np.sqrt(self.X.shape[1]))
        elif features_max_value_type == 'log2':
            self.n_features = int(np.log2(self.X.shape[1]))
        else:
            self.n_features = int(
                features_max_value_type(self.X.shape[1]))
        self.sampled_rows_indices, self.sampled_feature_indices = self._get_sampled_indices()
        self.trees = None

    def _get_sampled_indices(self):
        sampled_feature_indices = []
        sampled_rows_indices = []
        for _ in range(self.n_trees):
            np.random.seed(random_seed)
            sampled_feature_indices.append(
                np.random.permutation(self.X.shape[1])[:self.n_features])
            np.random.seed(random_seed)
            sampled_rows_indices.append(
                np.random.permutation(int(self.X.shape[0]*self.rows_percentage)))

        return sampled_rows_indices, sampled_feature_indices


class RandomForestClassifier(RandomForestBase):
    def __init__(self, X, y, n_trees, max_depth, features_max_value_type='sqrt'):
        super().__init__(X, y, n_trees, max_depth, features_max_value_type)
        self.trees = [DecisionTreeClassifier(
            max_depth=max_depth) for _ in range(self.n_trees)]

    def fit(self):
        for row_indices, feature_indices, tree in zip(self.sampled_rows_indices, self.sampled_feature_indices, self.trees):
            tree.fit(self.X[row_indices][:, feature_indices], self.y)

    def predict(self, X_test):
        predictions = []
        for feature_indices, tree in zip(self.sampled_feature_indices, self.trees):
            predictions.append(tree.predict(X_test[:, feature_indices]))
        predictions = stats.mode(np.array(predictions), axis=0)
        return predictions


class RandomForestRegressor(RandomForestBase):
    def __init__(self, X, y, n_trees, max_depth, features_max_value_type='sqrt'):
        super().__init__(X, y, n_trees, max_depth, features_max_value_type)
        self.trees = [DecisionTreeRegressor(
            max_depth=max_depth) for _ in range(self.n_trees)]

    def fit(self):
        for row_indices, feature_indices, tree in zip(self.sampled_rows_indices, self.sampled_feature_indices, self.trees):
            tree.fit(self.X[row_indices][:, feature_indices], self.y)

    def predict(self, X_test):
        predictions = []
        for feature_indices, tree in zip(self.sampled_feature_indices, self.trees):
            predictions.append(tree.predict(X_test[:, feature_indices]))        predictions = np.array(predictions).mean(0)
        return predictions
