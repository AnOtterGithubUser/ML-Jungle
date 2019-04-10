import numpy as np
from cvxopt import matrix, solvers
from utils.kernels import linear_kernel


class SVM:
    """ This class implements the binary support vector machines classifier.
    This method is based on solving the quadratic dual problem with cvxopt. Though it is faster than
    solving the primal, this is not the fastest method. Dual coordinate ascent can significantly
    speed this up.

    This class is implemented as an exercise. For a fast, scalable implementation use:
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """

    def __init__(self, C, kernel=linear_kernel):
        """
        Parameters
        ----------
        C: float
            Regularization parameter

        kernel: string or callable
            Specifies the kernel to use in the algorithm.
            It must take the data as input and return a matrix of size (n_samples, n_samples)
        """
        self.kernel = kernel
        self.C = C

    @staticmethod
    def _graam_matrix(X, kernel):
        K = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K[i, j] = kernel(X[i], X[j])
        return K

    def fit(self, X, y):
        n_samples, n_features = X.shape
        K = self._graam_matrix(X, self.kernel)
        P = matrix(np.outer(y, y) * K)
        q = matrix(np.ones(n_samples) * -1)
        G_0 = np.diag(np.ones(n_samples)) * -1
        G_C = np.diag(np.ones(n_samples))
        h_0 = np.zeros(n_samples)
        h_C = np.ones(n_samples) * self.C
        G = matrix(np.vstack((G_0, G_C)))
        h = matrix(np.vstack((h_0, h_C)))
        A = matrix(y, (1, n_samples))
        b = matrix(0.0)
        self.alpha_ = solvers.qp(P, q, G, h, A, b)
        return self

    def predict(self, X):
        raise NotImplementedError
