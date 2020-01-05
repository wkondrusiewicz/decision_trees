import os

import numpy as np
import pydot

from decision_trees.node import TreeNode
from decision_trees.metrics import GiniImportance


class BasicTree:
    def __init__(self, max_depth, scorer=GiniImportance):
        self.__max_depth = max_depth
        self.scorer = scorer()  # create an instance of scorer class
        self.tree = None

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

    def _get_best_split_for_feature(self, X_slice, y_slice):
        pass

    def _get_best_split_for_dataset(self, X, y):
        pass

    def _grow_tree(self, X, y, depth=0):
        pass


class DecisionTreeClassifier(BasicTree):
    def __init__(self, max_depth, scorer=GiniImportance):
        super().__init__(max_depth, scorer)

    def _get_best_split_for_feature(self, X_slice, y_slice):
        X_slice_sorted = np.sort(X_slice)
        thresholds = np.unique(
            (X_slice_sorted[1:] + X_slice_sorted[:-1]) / 2)
        current_fueature_score = self.scorer.get_score(y_slice)
        best_score = current_fueature_score
        best_threshold = None
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
        return best_score, best_index, best_thresh

    def _grow_tree(self, X, y, depth=0, tree_structure={}):
        current_score = self.scorer.get_score(y)
        tree_node = TreeNode(score=current_score)
        if depth < self.max_depth:
            best_score, best_index, best_thresh = self._get_best_split_for_dataset(X, y)
            if best_index is not None and best_thresh is not None:
                left_mask = X[:, best_index] < best_thresh
                right_mask = X[:, best_index] >= best_thresh
                X_left = X[left_mask]
                X_right = X[right_mask]
                y_left = y[left_mask]
                y_right = y[right_mask]
                tree_node.feature_index=best_index
                tree_node.threshold=best_thresh
                root_key = f'feature_{best_index}, thresh {best_thresh}'
                print(root_key)
                tree_structure[root_key]={}
                tree_node.tree_structure=tree_structure
                tree_node.left_node = self._grow_tree(X_left, y_left, depth+1, tree_structure[root_key])
                tree_node.right_node = self._grow_tree(X_right, y_right, depth+1, tree_structure[root_key])
        else:
            print('DUPA')
            tree_structure[f'TERMINAL NODE WITH SCORE {current_score}']={}
            tree_node.tree_structure = tree_structure
        return tree_node

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def draw_tree(self, out_dir):
        assert self.tree is not None, 'Please first fit a tree to the data'
        def draw(parent_name, child_name):
            edge = pydot.Edge(parent_name, child_name)
            graph.add_edge(edge)

        def visit(node, parent=None):
            for k,v in node.items():
                if isinstance(v, dict):
                    # We start with the root node whose parent is None
                    # we don't want to graph the None node
                    if parent:
                        draw(parent, k)
                    visit(v, k)
                else:
                    draw(parent, k)
                    # drawing the label using a distinct name
                    draw(k, k+'_'+v)

        graph = pydot.Dot(graph_type='graph')
        visit(self.tree.tree_structure)
        graph.write_png(os.path.join(out_dir,'decision_tree.png'))
