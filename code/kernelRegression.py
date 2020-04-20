import numpy as np
from numpy.linalg import solve, inv
import findMin
from scipy.optimize import approx_fprime
import utils


def log_1_plus_exp_safe(x):
    # compute log(1+exp(x)) in a numerically safe way, avoiding overflow/underflow issues
    out = np.log(1+np.exp(x))
    out[x > 100] = x[x>100]
    out[x < -100] = np.exp(x[x < -100])
    return out


def kernel_RBF(X1, X2, sigma=1):
    return np.exp(-1 * utils.euclidean_dist_squared(X1, X2) / (2*sigma**2))


def kernel_poly(X1, X2, p=2):
    return np.power(X1 @ X2.T + 1, p)


def kernel_linear(X1, X2):
    return X1@X2.T


class kernelLinearClassifier():
    def __init__(self, lammy=1.0, verbose=0, maxEvals=100, kernel_fun=kernel_RBF, **kernel_args):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals
        self.kernel_fun = kernel_fun
        self.kernel_args = kernel_args

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size
        self.X = X

        # Initial guess
        self.U = np.zeros((self.n_classes, n))

        K = self.kernel_fun(X,X, **self.kernel_args)

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y == i] = 1
            ytmp[y != i] = -1

            if ytmp.ndim == 1:
                ytmp = ytmp[:, None]

            U = inv(K + self.lammy*np.eye(n, n)) @ ytmp
            self.U[i] = U.T

    def predict(self, Xtest):
        Ktest = self.kernel_fun(Xtest, self.X, **self.kernel_args)
        return np.argmax(Ktest@self.U.T, axis=1)