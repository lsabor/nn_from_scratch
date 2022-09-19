"""
classes for a neural network
"""

from typing import Iterable
import numpy as np

from loss_functions import Loss
from weights import Transformation


class Network:
    """
    Neural Network class
    requires:
    - layer_sizes = an array of ints representing number of neurons in each hidden layer
    - Y = ...
    """

    initialized = False
    layer_descriptions = []

    def __init__(self, data_size, loss: Loss, layer_descriptions=None):
        """
        loss = a specific Loss object like CrossEntropy
        layer_descriptions = an iterable of tuples (activation, )

        """
        self.datapoint_size = data_size
        self.loss = loss

        if layer_descriptions:
            self.layer_descriptions = layer_descriptions
            self.initialize()

    def add_layer(self, layer_type, size):
        if self.initialized:
            raise Exception("cannot add layer once initialized")
        self.layer_descriptions.append((layer_type, size))

    def remove_layer(self, n=-1):
        if self.initialized:
            raise Exception("cannot remove layer once initialized")
        self.layer_descriptions.pop(n)

    def initialize(self):
        if self.initialized:
            raise Exception("already initialized")
        if not self.layer_descriptions:
            raise Exception("cannot initialize without layer descriptions")

        next_step = self.loss
        self.layers = [next_step]
        for layer in self.layer_descriptions[::-1]:
            current_layer = layer[0](size=layer[1], next_step=next_step)
            if isinstance(next_step, Loss):
                self.layers.append(current_layer)
                continue
            transformation = Transformation(prev=current_layer, next_step=next_step)
            self.layers.append(transformation)
            self.layers.append(current_layer)
        transformation = Transformation(
            prev=self.datapoint_size, next_step=current_layer.size
        )
        self.layers.append(transformation)

        self.initialized = True

    def forward_pass(self):
        ...
