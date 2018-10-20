import numpy as np
from abc import ABC, abstractmethod
from ml_lib.base.estimator import Estimator


class DecisionTree(Estimator):
    def __init__(self, max_depth=-1):
        self.max_depth = max_depth
        self._node = None

    def fit(self, X, y):
        self._node = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth:
            return self._create_leaf(y)
        split_rule = self._find_split_rule(X, y)
        if split_rule.information_gain > 0:
            node = self._create_branch(X, y, split_rule, depth=depth)
        else:
            node = self._create_leaf(y)
        return node

    def _create_leaf(self, y):
        return Leaf(value=self._voting(y))

    def _create_branch(self, X, y, split_rule, depth):
        mask = split_rule.condition(X[:, split_rule.feature_id])
        return Branch(
            split_rule=split_rule,
            left_node=self._build_tree(X[mask], y[mask], depth=depth + 1),
            right_node=self._build_tree(X[~mask], y[~mask], depth=depth + 1)
        )

    def _find_split_rule(self, X, y):
        feature_ids = np.arange(X.shape[1])
        split_rules = [self._find_split_rule_for_feature(X, y, feature_id) for feature_id in feature_ids]
        return max(split_rules, key=lambda split_rule: split_rule.information_gain)

    def _find_split_rule_for_feature(self, X, y, feature_id):
        X_i = X[:, feature_id]
        pivots = sorted(X_i)
        pivots += [X_i[-1] + 1]
        split_rules = []
        for pivot in pivots:
            condition = Condition(pivot=pivot)
            information_gain = self._compute_information_gain(X_i, y, condition=condition)
            split_rules.append(SplitRule(feature_id=feature_id, condition=condition, information_gain=information_gain))
        return max(split_rules, key=lambda split_rule: split_rule.information_gain)

    def predict(self, X):
        return np.asarray([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        node = self._node
        while not isinstance(node, Leaf):
            assert isinstance(node, Branch)
            split_rule = node.split_rule
            if split_rule.condition(x[split_rule.feature_id]):
                node = node.left_node
            else:
                node = node.right_node
        return node.value

    @abstractmethod
    def _voting(self, y):
        raise NotImplementedError()

    @abstractmethod
    def _compute_information_gain(self, X, y, condition):
        raise NotImplementedError()


class Condition(object):
    def __init__(self, pivot):
        self.pivot = pivot

    def __call__(self, X):
        return X < self.pivot


class SplitRule(object):
    def __init__(self, feature_id, condition, information_gain):
        self.feature_id = feature_id
        self.condition = condition
        self.information_gain = information_gain


class DecisionNode(ABC):
    pass


class Branch(DecisionNode):
    def __init__(self, split_rule, left_node, right_node):
        self.split_rule = split_rule
        self.left_node = left_node
        self.right_node = right_node


class Leaf(DecisionNode):
    def __init__(self, value):
        self.value = value
