"""
Loss functions for neural net
"""

import numpy as np


class Loss:
    """
    Loss function
    evaluates distance between prediction (Y_hat) and actual value (Y)
    """

    name = ...
    equation = ...
    out_shape = (1,)  # returns a constant

    def __init__(self, Y):
        self.Y = np.array(Y)

    def loss(self, Y_hat):
        ...


class CrossEntropy(Loss):
    """
    Y     = actual values
    Y_hat = estimate of Y

    L(Y,Y_hat) = - sum_over_i(Y_hat[i] * log(Y[i]))
    """

    name = "Cross Entropy"
    equation = "- sum_over_i(Y[i] * log(Y_hat[i]))"

    def loss(self, Y_hat):
        Y_hat_clipped = np.clip(Y_hat, 1e-7, 1)  # to remove errors when estimate is 0
        targeted_Y_hat = np.sum(Y_hat_clipped * self.Y, axis=1)
        return np.mean(-np.log(targeted_Y_hat))

    def deriv(self, Y_hat):
        """
        returns how self.loss changes with regards to a change in each value in Y_hat

        dL/dY_hat{n} = (- {sum over i of} (Y[i] / Y_hat[i])) / m
         = -np.mean(self.Y / Y_hat, axis=1)
        """
        return -np.mean(self.Y / Y_hat, axis=1)

    def deriv_loss(self, Y_hat):
        """
        since loss is it's return
        deriv_loss == deriv
        """
        return self.deriv(Y_hat)
