import numpy as np
from collections import Counter

from ml_lib.tree.decision_tree import DecisionTree
from ml_lib.viz_utils import plot_decision_boundary


def _compute_distribution(y):
    n = len(y)
    return [count / n for count in Counter(y).values()]


def _compute_entropy(y):
    return -sum([p*np.log2(p) for p in _compute_distribution(y)])


def _compute_gini(y):
    return 1 - sum(np.square(_compute_distribution(y)))


class DecisionTreeClassifier(DecisionTree):
    def __init__(self, max_depth=-1, criterion='entropy'):
        super().__init__(max_depth=max_depth)
        self.criterion = criterion
        self._criterion_fn = self._get_criterion_fn(criterion)

    def _compute_information_gain(self, X, y, condition):
        n = len(y)
        initial_criterion_value = self._criterion_fn(y)
        mask = condition(X)
        weight = mask.sum() / n
        return initial_criterion_value - weight*self._criterion_fn(y[mask]) - (1-weight)*self._criterion_fn(y[~mask])

    def _voting(self, y):
        return Counter(y).most_common(n=1)[0][0]

    def _get_criterion_fn(self, criterion):
        return {
            'entropy': _compute_entropy,
            'gini': _compute_gini
        }[criterion]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    X, y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0,
                               n_clusters_per_class=1, class_sep=0.2, random_state=17)
    estimator = DecisionTreeClassifier(max_depth=5, criterion='gini')
    train_predictions = estimator.fit_predict(X, y)
    print('train_accuracy: {}'.format(accuracy_score(y_true=y, y_pred=train_predictions)))
    plot_decision_boundary(X, y, estimator=estimator, num=100, figsize=(16, 16))
    plt.show()
