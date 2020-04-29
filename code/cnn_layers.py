# CITATION: Most of the code in this file is adapted from Mahan Fathi's Stanford's CS231 Assignment 2 repo
# <Mahan Fathi> (Sep, 2017) CS231/blob/master/assignment2/cs231n/layers.py.
#       https://github.com/MahanFathi/CS231/blob/master/assignment2/cs231n/layers.py

import numpy as np


def conv_layer_forward(x, w, b, stride, pad):
    '''
    Computes the layer which is the result of convolution between input x matrix and w filter
    :param x: Input of shape N x C x H x W
    :param w: Filters of shape F x C x HH x WW
    :param b: Bias for the filter of shape (F,)
    :param stride: integer value representing the stride of convolution filter
    :param pad: integer value representing number of zeros to be added along H and W axes of X
    :return: out: resulting layer formed after convolution of shape N x F x H_prime x W_prime
    :return: cache: cache is a tuple of x, w, b, stride, pad
    '''

    # Obtain the parameters from input matrices
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # Ensure that the filter dimensions and stride are compatible with input matrix dimensions
    assert (H - HH + 2*pad) % stride == 0, 'Failure: Filter height and stride incompatible with input matrix dimensions'
    assert (W - WW + 2*pad) % stride == 0, 'Failure: Filter width and stride incompatible with input matrix dimensions'

    # Add specified padding on x matrix
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)

    # Compute the output layer's height and width
    H_prime = 1 + (H - HH + 2*pad) // stride
    W_prime = 1 + (W - WW + 2*pad) // stride

    # Initialize the output matrix (it'll contain N values each with F layers, H_prime height, and W_prime height)
    out = np.zeros((N, F, H_prime, W_prime))

    # Compute the values of output matrix by convoluting the filter with input matrix
    for n in range(N):
        for f in range(F):
            for j in range(H_prime):
                for i in range(W_prime):
                    # Multiply each cell in the region occupied by the filter with the corresponding filter value, sum
                    # those multiplied values, and add filter value to the summed value
                    out[n, f, j, i] = (x_padded[n, :, j*stride:j*stride+HH, i*stride:i*stride+HH] * w[f,:,:,:]).sum() \
                                      + b[:, f]

    cache = (x, w, b, stride, pad)

    return out, cache


def conv_layer_backward(dupstream, cache):
    '''
    Computes the gradients dw, dw, and db using the gradient coming from upstream network (dupstream)
    :param dupstream: Gradient flowing from upstream the network
    :param cache: tuple containing x, w, b, stride, pad
    :return: dx: gradient with respect to x of shape N x C x H x W
    :return: dw: gradient with respect to w of shape F x C x HH x WW
    :return: db: gradient with respect to b of shape (F,)
    '''

    x, w, b, stride, pad = cache

    # Obtain the parameters from input matrices
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # Add specified padding on x matrix
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)

    # Compute the convolution filter output layer's height and width
    H_prime = 1 + (H - HH + 2 * pad) // stride
    W_prime = 1 + (W - WW + 2 * pad) // stride

    # Initialize the output matrices with zeros
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # Compute the gradients (these are derived by plotting the forward pass on a computational graph and applying
    # backpropagation techniques on the computational graph)
    for n in range(N):
        for f in range(F):
            # In computational graph, at an addition node, the upstream gradient is distributed to both input nodes
            # in this case, the input nodes are b and products of w and x
            db[:, f] += dupstream[n, f].sum()
            for j in range(0, H_prime):
                for i in range(0, W_prime):
                    # In computational graph, the gradients for inputs to multiplication node are simply the products of
                    # the other input value and upstream gradient
                    dw[f] += x_padded[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW] * dupstream[n, f, j, i]
                    dx_padded[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW] += w[f] * dupstream[n, f, j, i]

    # Extract the value of dx from dx_padded
    dx = dx_padded[:, :, pad:pad+H, pad:pad+W]

    return dx, dw, db


def relu_layer_forward(x):
    '''
    Computes the output matrix after applying RelU max(0, value) filter on input matrix x
    :param x: Input matrix
    :return: out: matrix of same dimensions as x containing values filtered by RelU
    :return: cache: contains the input matrix
    '''

    out = np.maximum(0, x)    # RelU filter
    cache = x

    return out, cache


def relu_layer_backward(dupstream, cache):
    '''
    Computes the gradient dx with respect to x using the upstream gradient, dupstream
    :param dupstream: Gradient of same dimensions as x flowing from upstream the network
    :param cache: contains the input matrix x
    :return: dx: gradient of the relU layer with respect to x
    '''
    x = cache

    # At the max node in a computational graph for RelU, the gradient only flows to the x input value if it is > 0
    dx = (x>0)*dupstream

    return dx


def max_pool_layer_forward(x, pool_height, pool_width, stride):
    '''
    Computes the output layer obtained after applying max_pool on the input layer x
    :param x: input layer of shape N x C x H x W
    :param pool_height: integer value representing the height of the max pool filter
    :param pool_width: integer value representing the width of the max pool filter
    :param stride: integer value representing the stride of max pool filter
    :return: out: layer obtained after max pooling the input layer of shape N x C x H_prime x W_prime
    :return: cache: tuple containing x, pool_height, pool_width, stride
    '''

    # Obtain the parameters from inputs
    N, C, H, W = x.shape
    HH = pool_height
    WW = pool_width

    # Check if the input matrix and max pool filter dimensions are compatible
    assert (H - HH) % stride == 0, 'Failure: Max Pool Filter''s height is not compatible with input layer'
    assert (W - WW) % stride == 0, 'Failure: Max Pool Filter''s width is not compatible with input layer'

    # Compute the output dimensions
    H_prime = 1 + (H-HH) // stride
    W_prime = 1 + (W-WW) // stride

    out = np.zeros((N, C, H_prime, W_prime))

    for n in range(N):
        for j in range(H_prime):
            for i in range(W_prime):
                out[n, :, j, i] = np.amax(x[n, :, j*stride:j*stride+HH, i*stride:i*stride+WW], axis=(-1, -2))

    cache = (x, pool_height, pool_width, stride)
    return out, cache


def max_pool_layer_backward(dupstream, cache):
    '''
    Computes the gradient of the max pooling layer with respect to x
    :param dupstream: Gradient of max pool layer from upstream the network
    :param cache: tuple containing x, pool_height, pool_width, stride
    :return: dx: gradient of max pool layer with respect to x
    '''

    # Obtain the parameters from inputs
    x, pool_height, pool_width, stride = cache
    HH = pool_height
    WW = pool_width
    N, C, H, W = x.shape

    # Compute the max pool output layer dimensions
    H_prime = 1 + (H - HH) // stride
    W_prime = 1 + (W - WW) // stride

    # Initialize the output gradient matrix with zeros
    dx = np.zeros_like(x)

    # The gradient will only flow back to the input cell which contributed to it, i.e. to the input cell which was the
    # max value in the space spanned by the filter
    for n in range(N):
        for c in range(C):
            for j in range(H_prime):
                for i in range(W_prime):
                    # Determine the index of the cell in x-space spanned by max pool filter containing the max value
                    ind = np.argmax(x[n, c, j * stride:j * stride + HH, i * stride:i * stride + WW])
                    HH_ind, WW_ind = np.unravel_index(ind, (HH, WW))
                    dx[n, c, j * stride:j * stride + HH, i * stride:i * stride + WW][HH_ind, WW_ind] = dupstream[n, c, j, i]

    return dx


def fully_connected_layer_forward(x, w, b):
    '''
    Computes the output layer based on the inputs x, w, and b (bias)
    :param x: Input layer of shape N x d1 x d2 x ... x dk which contains N examples with each example x[i] having shape
              d1 x d2 x ... x dk. This layer will be converted to a matrix of shape N x D
    :param w: Matrix of shape D x M where M is the number of units in the first fully connected layers
    :param b: Array of shape (M,) containing the bias values
    :return: out: output layer of shape N x M
    :return: cache: tuple containing x, w, b
    '''

    # Reshape the x layer to be a matrix of dimensions N x D and compute the output layer
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache


def fully_connected_layer_backward(dupstream, cache):
    '''
    Computes the gradients of the fully connected layer with respect to x, w, and b (bias)
    :param dupstream: gradient flowing from upstream the network and of the dimensions N x M, where M is the number of
                      hidden units in the fully connected layer
    :param cache: tuple containing x, w, b
    :return: dx: gradient with respect to x (N, D) or (N, d1, d2, ...., dk)
    :return: dw: gradient with respect to w (D, M)
    :return: db: gradient with respect to b (M,)
    '''

    # Obtain the matrices from inputs
    x, w, b = cache

    # These gradients are easy to determine if this simple X@W + b computation is plotted on a computation graph and
    # backpropagation technique is used
    dx = (dupstream@w.T).reshape(x.shape)        # resulting matrix is N x D or N x d1 x d2 x ... x dk depending on
                                                 # whether this layer was connected to a convolutional layer
    dw = x.reshape(x.shape[0], -1).T@dupstream   # resulting matrix is D x M
    db = np.sum(dupstream, axis=0)               # resulting array is of shape (M,)

    return dx, dw, db


def log_sum_exp(Z):
    Z_max = np.max(Z,axis=1)
    return Z_max + np.log(np.sum(np.exp(Z - Z_max[:,None]), axis=1)) # per-colmumn max


def softmax_loss(x, y):
    '''
    Computes the softmax loss and gradient for the final softmax classification layer
    :param x: Prediction matrix of shape N x C, where C is the number of classes to be predicted from
    :param y: Matrix of shape N x C, which contains value of 1 (or True) in the correct class column of each example
    :return: loss: scalar value of the softmax loss
    :return: dx: gradient with respect to the input matrix x
    '''
    # shifted_logits = x - np.max(x, axis=1, keepdims=True)
    # Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    # log_probs = shifted_logits - np.log(Z)
    # probs = np.exp(log_probs)
    # N = x.shape[0]
    # loss = -np.sum(log_probs[np.arange(N), y]) / N
    # dx = probs.copy()
    # dx[np.arange(N), y] -= 1
    # dx /= N
    # return loss, dx

    tmp = np.sum(np.exp(x), axis=1)
    loss = -np.sum(x[y.astype(bool)] - log_sum_exp(x))
    dx = np.exp(x) / tmp[:, None] - y

    return loss, dx
