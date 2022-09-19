"""
Module for creating weights and biases for neural net
"""

import numpy as np


class Weights:
    """
    Weights applied between neural layers
    Notation: input = A, and output = Z* (Z without Biases)
    """

    name = "Weights"
    equation = "W dot A"

    def __init__(self, prev, next):
        self.next = next

        self.W = np.random.rand((next.shape[0], prev.shape[-1]))
