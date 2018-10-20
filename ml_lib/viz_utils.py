import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(X, y, estimator, num=100, ax=None, cmap='autumn', **fig_kwargs):
    X, y = np.asarray(X), np.asarray(y)
    assert X.ndim >= 2
    x0s = np.linspace(X[:, 0].min(), X[:, 0].max(), num=num)
    x1s = np.linspace(X[:, 1].min(), X[:, 1].max(), num=num)
    x0s, x1s = np.meshgrid(x0s, x1s)
    X_mesh = np.c_[x0s.ravel(), x1s.ravel()]
    predictions = estimator.predict(X_mesh).reshape(x0s.shape)
    if ax is None:
        _, ax = plt.subplots(**fig_kwargs)
    ax.pcolormesh(x0s, x1s, predictions, cmap=cmap)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black', cmap=cmap)
    return ax
