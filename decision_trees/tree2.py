import os

import numpy as np
import pydot

from decision_trees.node import TreeNode
from decision_trees.metrics import GiniImportance, StandardDeviationReduction


class BasicTreePredictor:
    def __init__(self, max_depth, scorer, mode):
        assert mode in ['classification',
                        'regression'], 'Invalid mode type'
        self.__max_depth = max_depth
        self.scorer = scorer()  # create an instance of scorer class
        self.mode = mode
        self.tree = None
        self.number_of_classes = None

    @property
    def max_depth(self):
        return self.__max_depth

    @max_depth.setter
    def max_depth(self, max_depth):
        self.__max_depth = max_depth

    def _get_best_split_for_feature(self, X_slice, y_slice):
        X_slice_sorted = np.sort(X_slice)
        thresholds = np.unique(
            (X_slice_sorted[1:] + X_slice_sorted[:-1]) / 2)
        current_fueature_score = self.scorer.get_score(y_slice)
        best_score = current_fueature_score
        best_threshold = None
        if len(y_slice) <= 1:  # >= 2 elements are needed to split a Node
            return None, None
        for thresh in thresholds:
            score = self.scorer.get_score_after_split(
                X_slice, y_slice, thresh)
            if score < best_score:
                best_score = score
                best_threshold = thresh
        return best_score, best_threshold

    def _get_best_split_for_dataset(self, X, y):
        current_fueature_score = self.scorer.get_score(y)
        best_score = current_fueature_score
        for i in range(X.shape[1]):
            X_slice = X[:, i]
            best_score_for_feature, best_thresh_for_feature = self._get_best_split_for_feature(
                X_slice, y)
            if best_thresh_for_feature is not None and self.scorer.scoring_condition(best_score_for_feature, best_score):
                best_score = best_score_for_feature
                best_index = i
                best_thresh = best_thresh_for_feature
            elif best_thresh_for_feature is None:
                best_index = None
                best_thresh = None
        return best_score, best_index, best_thresh

    def _grow_tree(self, X, y, depth=0, tree_structure={}):
        if depth == 0 and len(tree_structure) == 0:
            tree_structure = {}

        current_score = self.scorer.get_score(y)
        if self.mode == 'classification':
            samples_per_class = [np.sum(y == i)
                                 for i in range(1, self.number_of_classes + 1)]
            prediction = np.argmax(samples_per_class) + 1
        elif self.mode == 'regression':
            samples_per_class = None
            prediction = np.mean(y)

        tree_node = TreeNode(
            score=current_score, samples_per_class=samples_per_class, prediction=prediction)
        if depth < self.max_depth:
            best_score, best_index, best_thresh = self._get_best_split_for_dataset(
                X, y)
            if best_index is not None and best_thresh is not None:
                left_mask = X[:, best_index] < best_thresh
                right_mask = X[:, best_index] >= best_thresh
                X_left = X[left_mask]
                X_right = X[right_mask]
                y_left = y[left_mask]
                y_right = y[right_mask]
                tree_node.feature_index = best_index
                tree_node.threshold = best_thresh
                root_key = self.get_output_format(
                    best_score, samples_per_class, prediction, best_index, best_thresh)
                tree_structure[root_key] = {}
                tree_node.tree_structure = tree_structure
                tree_node.left_node = self._grow_tree(
                    X_left, y_left, depth + 1, tree_structure[root_key])
                tree_node.right_node = self._grow_tree(
                    X_right, y_right, depth + 1, tree_structure[root_key])
        else:
            root_key = self.get_output_format(
                current_score, samples_per_class, prediction)
            tree_structure[root_key] = {}
            tree_node.tree_structure = tree_structure
        return tree_node

    def get_output_format(self, score, samples_per_class, prediction, index=None, threshold=None):
        root_key = f'{type(self.scorer).__name__} = {score:.2f}\n'
        if self.mode == 'classification':
            root_key += f'samples = {np.sum(samples_per_class)}\nclass division = {samples_per_class}\n'
        root_key += f'prediction = {prediction}'
        if threshold is not None and index is not None:
            root_key = f'feature_{index} < {threshold:.2f}\n' + root_key
        return root_key

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _get_single_prediction(self, X_single):
        assert self.tree is not None, 'Please first fit a tree to the data'

        node = self.tree
        while node.left_node:
            if X_single[node.feature_index] < node.threshold:
                node = node.left_node
            else:
                node = node.right_node
        return node.prediction

    def predict(self, X):
        return [self._get_single_prediction(x) for x in X]

    def draw_tree(self, out_dir):
        assert self.tree is not None, 'Please first fit a tree to the data'

        graph = pydot.Dot(graph_type='graph')

        def draw(parent_name, child_name):
            edge = pydot.Edge(parent_name, child_name)
            graph.add_edge(edge)

        def visit(node, parent=None):
            for k, v in node.items():
                if isinstance(v, dict):
                    # We start with the root node whose parent is None
                    # we don't want to graph the None node
                    if parent:
                        draw(parent, k)
                    visit(v, k)
                else:
                    draw(parent, k)
                    # drawing the label using a distinct name
                    draw(k, k + '_' + v)

        visit(self.tree.tree_structure)
        os.makedirs(out_dir, exist_ok=True)
        graph.write_png(os.path.join(
            out_dir, f'decision_tree_{self.mode}.png'))


class DecisionTreeClassifier(BasicTreePredictor):
    def __init__(self, max_depth, scorer=GiniImportance):
        super().__init__(max_depth, scorer,  mode='classification')
        assert self.mode in self.scorer.optimization_types, f'scorer {self.scorer} is incompatible with this predictor'

    def fit(self, X, y):
        self.number_of_classes = len(set(y))
        super().fit(X, y)


class DecisionTreeRegressor(BasicTreePredictor):
    def __init__(self, max_depth, scorer=StandardDeviationReduction):
        super().__init__(max_depth, scorer, mode='regression')
        assert self.mode in self.scorer.optimization_types, f'scorer {self.scorer} is incompatible with this predictor'
