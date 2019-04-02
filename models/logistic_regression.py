import numpy as np
import warnings


class BinaryLogisticRegression:
    """This class implements the classic binary logistic regression
    with vanilla gradient ascent to optimize the log likelihood.
    It follows the scikit learn model, most attributes and methods have the
    same name and usage.
    Its purpose is learning to code classic machine learning
    algorithm from scratch. If you need a fast, scalable, multi-class implementation
    of logistic regression, I would recommend using the scikit learn implementation:
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    def __init__(self, tolerance=1e-5, with_bias=True, max_iter=1000, learning_rate=1e-4):
        """
        Parameters
        ----------
        tolerance: float
            tolerance criterion to stop optimizing

        with_bias: bool
            If True, add an intercept to the model.
            Otherwise, it assumes that your data are already centered

        max_iter: int
            Maximum number of iterations while optimizing

       learning_rate: float
            Learning rate of the optimization algorithm

        """
        self.tolerance = tolerance
        self.with_bias = with_bias
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def add_bias(X):
        return np.hstack((X, np.ones((X.shape[0], 1))))

    def fit(self, X, y):
        """Fit the parameters of the model on the given data set
        Parameters
        ----------
        X: numpy.ndarray
            Feature matrix of size Nxp

        y: numpy.ndarray
            Labels

        """
        if self.with_bias:
            X = self.add_bias(X)
        self.param = np.zeros(X.shape[1])
        for step in range(self.max_iter):
            y_pred = self.sigmoid(np.dot(X, self.param))
            gradient = np.dot(X.T, y - y_pred)
            if np.linalg.norm(self.learning_rate * gradient) < self.tolerance:
                break
            self.param += self.learning_rate * gradient
        return self

    def predict_proba(self, X):
        if self.with_bias:
            X = self.add_bias(X)
        return self.sigmoid(np.dot(X, self.param))

    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas > 0.5).astype(int)
