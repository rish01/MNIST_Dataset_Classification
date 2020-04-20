import numpy as np
from numpy.linalg import norm
from math import sqrt

def findMin(funObj, w, maxEvals, *args, verbose=0):
    """
    Uses gradient descent to optimize the objective function

    This uses quadratic interpolation in its line search to
    determine the step size alpha
    """
    # Parameters of the Optimization
    optTol = 1e-2
    gamma = 1e-4

    # Evaluate the initial function value and gradient
    f, g = funObj(w,*args)
    funEvals = 1

    alpha = 1.
    while True:
        # Line-search using quadratic interpolation to find an acceptable value of alpha
        gg = g.T@g

        while True:
            w_new = w - alpha * g
            f_new, g_new = funObj(w_new, *args)

            funEvals += 1

            if f_new <= f - gamma * alpha*gg:
                break

            if verbose > 1:
                print("f_new: %.3f - f: %.3f - Backtracking..." % (f_new, f))

            # Update step size alpha
            alpha = (alpha**2) * gg/(2.*(f_new - f + alpha*gg))

        # Print progress
        if verbose > 0:
            print("%d - loss: %.3f" % (funEvals, f_new))

        # Update step-size for next iteration
        y = g_new - g
        alpha = -alpha*(y.T@g) / (y.T@y)

        # Safety guards
        if np.isnan(alpha) or alpha < 1e-10 or alpha > 1e10:
            alpha = 1.

        if verbose > 1:
            print("alpha: %.3f" % (alpha))

        # Update parameters/function/gradient
        w = w_new
        f = f_new
        g = g_new

        # Test termination conditions
        optCond = norm(g, float('inf'))

        if optCond < optTol:
            if verbose:
                print("Problem solved up to optimality tolerance %.3f" % optTol)
            break

        if funEvals >= maxEvals:
            if verbose:
                print("Reached maximum number of function evaluations %d" % maxEvals)
            break

    return w, f


def findMinSGD(funObj, w, epoch, minibatch_size, X, y, verbose=0):
    """
    Uses gradient descent to optimize the objective function

    This uses quadratic interpolation in its line search to
    determine the step size alpha
    """
    # Parameters of the Optimization
    num_iterations = X.shape[0]/minibatch_size * epoch
    alpha = 0.001

    # Obtain the initial random minibatch from the training set
    minibatch_indices = np.random.choice(X.shape[0], size=minibatch_size, replace=False)
    X_minibatch = X[minibatch_indices, :]
    y_minibatch = y[minibatch_indices, :]

    # Evaluate the initial function value and gradient
    f, g = funObj(w, X_minibatch, y_minibatch)
    funEvals = 1

    while True:
        # Update w_new
        w_new = w - alpha * g

        minibatch_indices = np.random.choice(X.shape[0], size=minibatch_size, replace=False)
        X_minibatch = X[minibatch_indices, :]
        y_minibatch = y[minibatch_indices, :]
        f_new, g_new = funObj(w_new, X_minibatch, y_minibatch)
        funEvals += 1
        alpha = 0.001/sqrt(funEvals)    # as recommended in lecture slides
        # alpha = 0.001/(1+funEvals)    # as recommended in lecture slides

        # Print progress
        if verbose > 0:
            print("%d - loss: %.3f" % (funEvals, f_new))

        if verbose > 1:
            print("alpha: %.7f" % (alpha))

        # Update parameters/function/gradient
        w = w_new
        f = f_new
        g = g_new

        # Test termination condition
        if funEvals >= num_iterations:
            if verbose:
                print("Reached maximum number of function evaluations %d" % num_iterations)
            break

    return w, f


