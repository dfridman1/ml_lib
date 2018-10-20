import numpy as np
from ml_lib.tree.decision_tree import DecisionTree


class DecisionTreeRegressor(DecisionTree):
    def _voting(self, y):
        return np.mean(y)

    def _compute_information_gain(self, X, y, condition):
        criterion_value = self.criterion_fn(y)
        mask = condition(X)
        return criterion_value - self.criterion_fn(y[mask]) - self.criterion_fn(y[~mask])

    def criterion_fn(self, y):
        y_mean = np.mean(y)
        return np.square(y - y_mean).sum()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    f = lambda x: np.exp(-x*x) + 1.5*np.exp(-(x-2)**2)
    num = 1000
    xs = np.linspace(-5, 5, num=num)
    ys = f(xs)
    plt.plot(xs, ys)
    noise = np.random.normal(loc=0, scale=.1, size=num)
    ys = ys + noise
    plt.scatter(xs[::10], ys[::10])
    estimator = DecisionTreeRegressor(max_depth=4)
    y_pred = estimator.fit_predict(xs.reshape(-1, 1), ys)
    plt.plot(xs, y_pred)
    plt.show()
