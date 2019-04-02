import numpy as np
import warnings


class BinaryLinearRegression:
    """This class implements the basic linear regression
    The sklearn implementation is already very simple, basically
    a wrapper around scipy's least square regression.
    For this class I wanted to use the normal equation instead of
    scipy.linalg.lstsq. There are issues when X^T X is not inversible.
    In "The Elements of Statistical Learning, Hastie, Tibshirani, Friedman, Springer, 2008",
    they recommend dropping the redundant columns of X so it has full rank and X^T X is inversible.
    I used the Moore-Penrose pseudo-inverse for the sake of simplicity. It already has an implementation
    in numpy.

    Like the logistic regression, the only purpose of this class is to
    implement classic machine learning algorithms from scratch. For a
    fast, scalable, multi-class implementation of the linear regression
    I would recommend using the scikit learn implementation:
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    def __init__(self, add_bias=True):
        """
        Parameters
        ----------
        add_bias: bool
            If True, add an intercept to the model.
            Otherwise, it assumes that your data are already centered

        """
        self.add_bias = add_bias

    def fit(self, X, y):
        if self.add_bias:
            X = np.hstack((X, np.ones(X.shape[0], 1)))
        if np.linalg.matrix_rank(np.dot(X.T, X)) == X.shape[1]:
            self.theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        else:
            warnings.warn("The graam matrix is not inversible, we use the Moore Penrose pseudo-inverse in that case")
            self.theta = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
        return self

    def predict_proba(self, X):
        if self.add_bias:
            X = np.hstack((X, np.ones(X.shape[0], 1)))
        return np.dot(X, self.theta)

    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas > 0.5).astype(int)

