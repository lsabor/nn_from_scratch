"""
classes for a neural network
"""

from typing import Iterable
import numpy as np
from activation_functions import ActivationLayer

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

    def __init__(self, data, Y, Loss, layer_descriptions=None):
        """
        loss = a specific Loss object like CrossEntropy
        layer_descriptions = an iterable of tuples (activation, )

        """
        self.X = data
        self.m = data.shape[-1]
        self.Loss = Loss(Y=Y)

        self.layer_descriptions = []
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

        next_step = self.Loss
        self.layers = [next_step]
        for layer in self.layer_descriptions[::-1]:
            current_layer = layer[0](size=layer[1], next_step=next_step)
            if isinstance(next_step, Loss):
                self.layers.append(current_layer)
                next_step = current_layer
                continue
            transformation = Transformation(
                prev_size=current_layer.size, next_step_size=next_step.size
            )
            self.layers.append(transformation)
            self.layers.append(current_layer)
            next_step = current_layer
        transformation = Transformation(
            prev_size=self.X.shape[0], next_step_size=current_layer.size
        )
        self.layers.append(transformation)

        self.initialized = True

    def forward_pass(self):
        current = self.X
        vals = [current]
        for layer in self.layers[:0:-1]:
            new_current = layer.apply(current)
            vals.append(new_current)
            current = new_current
        loss = self.Loss.loss(new_current)

        return (vals, loss)

    def backwards_propagation(self, vals):
        """
        vals[-1] = Y_hat
        vals[-2] = Z_final
        vals[-3] = A_final
        ...
        vals[3] = Z1
        vals[2] = A1
        vals[1] = Z0
        vals[0] = data
        """
        # DLs = []
        dWs = []
        dbs = []
        last_layer = self.layers[1]
        DZ = last_layer.deriv_loss(vals[-1])
        # DLs.append(DZ)

        for index, layer in enumerate(self.layers[2:]):

            if isinstance(layer, Transformation):
                A = vals[-index - 3]
                DA = layer.deriv_loss(DZ)
                DW = layer.weights.deriv_loss(m=self.m, A=A, DZ=DZ)
                Db = layer.biases.deriv_loss(m=self.m, DZ=DZ)

                # DLs.append(DA)
                dWs.append(DW)
                dbs.append(Db)

            elif isinstance(layer, ActivationLayer):
                Z = vals[-index - 3]
                dZ = layer.deriv(Z)
                DZ = layer.deriv_loss(DA=DA, dZ=dZ)
                # DLs.append(DZ)

        return (dWs, dbs)

    def update(self, lr, dWs, dbs):
        i = 0
        for layer in self.layers:
            if isinstance(layer, Transformation):
                layer.weights.W -= lr * dWs[i]
                layer.biases.b -= lr * dbs[i]
                i += 1
