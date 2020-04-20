"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
        X = self.X
        k = self.k
        y = self.y

        y_pred = np.zeros(Xtest.shape[0])
        euclidean_distances = np.argsort(utils.euclidean_dist_squared(Xtest, X))
        for n in range(Xtest.shape[0]):
            y_pred[n] = utils.mode(y[euclidean_distances[n, 0:k]])
        return y_pred
