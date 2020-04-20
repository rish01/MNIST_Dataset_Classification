import numpy as np

from decorators import time_me
from findMin import findMinSGD


class SVM_Multiclass_Sum_Loss():
    def __init__(self, lammy=1, epoch=1, verbose=1):
        self.lammy = lammy
        self.epoch = epoch
        self.verbose = verbose

    @time_me
    def fit(self, X, y):
        n, d = X.shape
        k = np.unique(y).size

        if y.ndim == 1:
            y = y[:, None]

        self.n_classes = k
        self.W = np.zeros(self.n_classes * d)

        self.W, f = findMinSGD(self.func_obj, self.W, epoch=self.epoch, minibatch_size=1, X=X, y=y, verbose=self.verbose)
        self.W = np.reshape(self.W, (self.n_classes, d))

    def func_obj(self, W, x_i, y_i):
        n, d = x_i.shape
        W = np.reshape(W, (self.n_classes, d))
        y_i = np.asscalar(y_i)

        # y_vals = np.argmax(y, axis=1)
        Wx_i = W@x_i.T

        # Compute the function value
        maxima = np.maximum(np.zeros((self.n_classes,1)), 1+Wx_i-Wx_i[y_i])
        maxima[y_i] = 0
        f = np.sum(maxima) + self.lammy/2 * np.linalg.norm(W) ** 2

        # Compute the gradient value
        g = np.zeros((self.n_classes, d))
        I = (1. + Wx_i - Wx_i[y_i]) > 0
        I[y_i] = 0

        for c in range(self.n_classes):
            g[c, :] = I[c] * x_i

        g[y_i, :] = -np.sum(I) * x_i
        g = g + self.lammy * W

        return f, g.flatten()

    def predict(self, X):
        return np.argmax(X @ self.W.T, axis=1)


class SVM_Multiclass_Max_Loss(SVM_Multiclass_Sum_Loss):
    def func_obj(self, W, x_i, y_i):
        n, d = x_i.shape
        W = np.reshape(W, (self.n_classes, d))
        y_i = np.asscalar(y_i)

        # y_vals = np.argmax(y, axis=1)
        Wx_i = W@x_i.T

        # Compute the function value
        maxima = np.maximum(np.zeros((self.n_classes,1)), 1+Wx_i-Wx_i[y_i])
        maxima[y_i] = 0
        f = np.max(maxima) + self.lammy/2 * np.linalg.norm(W) ** 2
        max_c = np.argmax(maxima)

        # Compute the gradient value
        g = np.zeros((self.n_classes, d))
        I = ((1. + Wx_i[max_c] - Wx_i[y_i]) > 0).astype(int)

        g[max_c, :] = I * x_i
        g[y_i, :] = -I * x_i

        g = g + self.lammy * W

        return f, g.flatten()
