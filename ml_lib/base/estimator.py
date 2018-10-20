from abc import ABC, abstractmethod


class Estimator(ABC):
    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()
