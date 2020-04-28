# CITATION: The structure of BasicCNN class code is adapted from Mahan Fathi's Stanford's CS231 Assignment 2 repo
# <Mahan Fathi> (Sep, 2017) CS231/blob/master/assignment2/cs231n/classifiers/cnn.py.
#       https://github.com/MahanFathi/CS231/blob/master/assignment2/cs231n/classifiers/cnn.py

from cnn_layers import *
import numpy as np
from findMin import findMinSGD


def tuple_product(layer_size_tuple):
    prod = 1
    for i in layer_size_tuple:
        prod = prod * i
    return prod


def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in sum(weights,())])


def unflatten_weights(weights_flat, layer_sizes):
    weights = list()
    counter = 0
    for i in range(len(layer_sizes)):
        W_size = tuple_product(layer_sizes[i][0])
        b_size = tuple_product(layer_sizes[i][1])

        W = np.reshape(weights_flat[counter:counter + W_size], layer_sizes[i][0])
        counter += W_size

        b = weights_flat[counter:counter + b_size][None]
        counter += b_size

        weights.append((W, b))
    return weights


class BasicCNN:
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - affine - relu - affine - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dim=(1, 28, 28),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 lammy=0.0,
                 conv_filter_stride=1,
                 max_pool_size=2,
                 max_pool_stride=2,
                 epoch=1,
                 minibatch_size=100,
                 verbose=0,
                 learning_rate_decay=False
                 ):
        self.params = {}
        self.lammy = lammy
        self.conv_filter_stride = conv_filter_stride
        self.max_pool_size = max_pool_size
        self.max_pool_stride = max_pool_stride
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.hidden_dim = hidden_dim
        self.epoch = epoch
        self.minibatch_size = minibatch_size
        self.verbose = verbose
        self.learning_rate_decay = learning_rate_decay

    def fit(self, X, y):
        if y.ndim == 1:
            y = y[:, None]

        self.num_classes = y.shape[1]

        C, H, W = self.input_dim
        first_fc_layer_height = self.num_filters*(((self.input_dim[1] - self.filter_size)/self.conv_filter_stride + 1)/self.max_pool_size)**2
        first_fc_layer_height = int(first_fc_layer_height)

        self.layer_sizes = [[(self.num_filters, C, self.filter_size, self.filter_size), (self.num_filters,)],
                            [(first_fc_layer_height, self.hidden_dim), (self.hidden_dim,)],
                            [(self.hidden_dim, self.num_classes), (self.num_classes,)]]

        # random init
        scale = 0.01
        self.params['W1'] = scale * np.random.randn(self.num_filters, C, self.filter_size, self.filter_size)
        self.params['b1'] = np.zeros(self.num_filters)

        self.params['W2'] = scale * np.random.randn(first_fc_layer_height, self.hidden_dim)
        self.params['b2'] = np.zeros(self.hidden_dim)
        self.params['W3'] = scale * np.random.randn(self.hidden_dim, self.num_classes)
        self.params['b3'] = np.zeros(self.num_classes)

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        weights = [(W3, b3), (W2, b2), (W1, b1)]
        weights_flat = flatten_weights(weights)
        weights_flat_new, f = findMinSGD(self.funObj, weights_flat,
                                         epoch=self.epoch,
                                         minibatch_size=self.minibatch_size,
                                         X=X, y=y,
                                         verbose=self.verbose,
                                         learning_rate_decay=self.learning_rate_decay)

        self.weights = unflatten_weights(weights_flat_new, self.layer_sizes)

    def funObj(self, weights_flat, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        """
        # Unflatten incoming flattened weights
        g = unflatten_weights(weights_flat, self.layer_sizes)

        # Reshape incoming input X images
        N = X.shape[0]
        X = X.flatten()
        X = X.reshape((N,)+self.input_dim)

        # Obtain the weights and biases from unflattened gradient
        W1, b1 = g[0]
        W2, b2 = g[1]
        W3, b3 = g[2]

        # Forward Pass: conv - relu - 2x2 max pool - fully connected - relu - fully connected - softmax
        out1, cache1 = conv_layer_forward(x=X, w=W1, b=b1, stride=self.conv_filter_stride, pad=0)
        out2, cache2 = relu_layer_forward(x=out1)
        out3, cache3 = max_pool_layer_forward(x=out2, pool_height=self.max_pool_size, pool_width=self.max_pool_size,
                                              stride=self.max_pool_stride)
        out4, cache4 = fully_connected_layer_forward(x=out3, w=W2, b=b2)
        out5, cache5 = relu_layer_forward(x=out4)
        out6, cache6 = fully_connected_layer_forward(x=out5, w=W3, b=b3)
        Z = out6

        f, grads = 0, {}

        # Backward Pass
        f, dout = softmax_loss(Z, y)
        dout, grads['W3'], grads['b3'] = fully_connected_layer_backward(dupstream=dout, cache=cache6)
        dout = relu_layer_backward(dupstream=dout, cache=cache5)
        dout, grads['W2'], grads['b2'] = fully_connected_layer_backward(dupstream=dout, cache=cache4)
        dout = max_pool_layer_backward(dupstream=dout, cache=cache3)
        dout = relu_layer_backward(dupstream=dout, cache=cache2)
        _, grads['W1'], grads['b1'] = conv_layer_backward(dupstream=dout, cache=cache1)

        g = [(grads['W3'], grads['b3']), (grads['W2'], grads['b2']), (grads['W1'], grads['b1'])]
        g = flatten_weights(g)

        # add L2 regularization
        f += 0.5 * self.lammy * np.sum(weights_flat ** 2)
        g += self.lammy * weights_flat

        return f, g

    def predict(self, X):

        W1, b1 = self.weights[0]
        W2, b2 = self.weights[1]
        W3, b3 = self.weights[2]

        # Forward Pass: conv - relu - 2x2 max pool - fully connected - relu - fully connected - softmax
        out1, cache1 = conv_layer_forward(x=X, w=W1, b=b1, stride=self.max_pool_stride, pad=0)
        out2, cache2 = relu_layer_forward(x=out1)
        out3, cache3 = max_pool_layer_forward(x=out2, pool_height=self.max_pool_size, pool_width=self.max_pool_size,
                                              stride=self.max_pool_stride)
        out4, cache4 = fully_connected_layer_forward(x=out3, w=W2, b=b2)
        out5, cache5 = relu_layer_forward(x=out4)
        out6, cache6 = fully_connected_layer_forward(x=out5, w=W3, b=b3)
        Z = out6

        return np.argmax(Z, axis=1)