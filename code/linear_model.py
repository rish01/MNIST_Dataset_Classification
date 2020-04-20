import numpy as np
from numpy.linalg import solve

from findMin import findMinSGD


def log_sum_exp(Z):
    Z_max = np.max(Z,axis=1)
    return Z_max + np.log(np.sum(np.exp(Z - Z_max[:,None]), axis=1))      # per-column max


# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        # self.w = lstsq(X.T@X, X.T@y)[0]
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w


class LeastSquaresL2(LeastSquares):
    def fit(self, X, y, lammy):
        n, d = X.shape
        self.w = solve(X.T@X + lammy*np.eye(d, d), X.T@y)


class leastSquaresClassifier:
    def __init__(self, lammy):
        self.lammy = lammy

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+self.lammy*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)


class LinearClassifierRobust(leastSquaresClassifier):
    def fit(self,X,y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes, d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y == i] = 1
            ytmp[y != i] = -1

            if ytmp.ndim == 1:
                ytmp = ytmp[:, None]

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            W, f = findMinSGD(self.funObj, self.W[i][:, None], epoch=1, minibatch_size=500, X=X, y=ytmp, verbose=0)
            self.W[i] = W.T

    def funObj(self,w,X,y):
        if w.ndim == 1:
            w = w[:, None]

        ''' MODIFY THIS CODE '''
        # Calculate the function value
        f = np.sum(np.log(np.exp(X @ w - y) + np.exp(y - X @ w)))

        # Calculate the gradient value
        g = X.T @ ((np.exp(X @ w - y) - np.exp(y - X @ w)) / (np.exp(X @ w - y) + np.exp(y - X @ w)))

        return (f,g)