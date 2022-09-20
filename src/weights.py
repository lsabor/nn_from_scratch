"""
Module for creating weights and biases for neural net
"""

import numpy as np


class Weights:
    """
    Weights applied between neural layers
    Notation: input = A, and output = Zw (Z without Biases)
    """

    name = "Weights"
    equation = "W dot A"

    def __init__(self, prev_size, next_size):
        self.W = np.random.randn(next_size, prev_size)

    def apply(self, A):
        """
        returns Zw from equation
        Z = Zw + b = W dot A + b
        """
        return np.dot(self.W, A)

    def deriv(self, A):
        """
        finding dZ2/dw2{n,n}:
        from (9)     Z2{n,m} = W2{n,n} dot A1{n,m} + b2{n,1}
        let i,j,k in range(n), dropping m for now
        Z2[i] = W2[i]{n} dot A1{n} + b2[i] = {sum over j} W2[i][j] * A1[j] + b2[i]
        dZ2[i]/dW2[j,k]{1} = 0 if i != j, else A1[k]
        dZ2[i]/dW2[i,k]{1} = A1[k]
        dZ2/dW2{n} = A1
        adding m back in: for l in range(m)
        Z2[i,l]{1} = W2[i] dot A1[l] + b2[i]
        dZ2[i,l]/dW2[i,k]{1} = A1[k,l]
        dZ2[l]/dW2{n} = A1[l]{n}
        dZ2/dW2{n,m} = A1{n,m}
        This shows what you would multiply a delta_W with to get the difference in Z2
        had you added that delta_W to W2 and recalulated Z2 that way
        """
        return A

    def deriv_loss(self, m, A, DZ):
        """
        DW2{n,n} = DZ2 * dZ2/dW2 = DZ2{n} dot A1{n}
        The derivative of the loss with respect to particular values of W2
        To bring m back in the picture, we have to average over all of the losses accrued
        during the training run. Namely m training examples:
        (15)    DW2{n,n} = 1/m * DZ2 * dZ2/dw2 = 1/m * DZ2{n,m} dot A1{n,m}.T{m,n}
        """
        return 1 / m * np.dot(DZ, A.T)


class Biases:
    """
    Biases applied between neural layers
    Notation: input = Zw, and output = Z
    """

    name = "Biases"
    equation = "Zw + b"

    def __init__(self, next_size):
        self.b = np.random.randn(next_size, 1)

    def apply(self, Zw):
        """
        returns Zw from equation
        Z = Zw + b = W dot A + b
        """
        return Zw + self.b

    def deriv(self):
        """
        finding dZ2/dw2{n,n}:
        from (9)     Z2{n,m} = W2{n,n} dot A1{n,m} + b2{n,1}
        let i,j,k in range(n), dropping m for now
        Z2[i] = W2[i]{n} dot A1{n} + b2[i] = {sum over j} W2[i][j] * A1[j] + b2[i]
        dZ2[i]/dW2[j,k]{1} = 0 if i != j, else A1[k]
        dZ2[i]/dW2[i,k]{1} = A1[k]
        dZ2/dW2{n} = A1
        adding m back in: for l in range(m)
        Z2[i,l]{1} = W2[i] dot A1[l] + b2[i]
        dZ2[i,l]/dW2[i,k]{1} = A1[k,l]
        dZ2[l]/dW2{n} = A1[l]{n}
        dZ2/dW2{n,m} = A1{n,m}
        This shows what you would multiply a delta_W with to get the difference in Z2
        had you added that delta_W to W2 and recalulated Z2 that way
        """
        return 1

    def deriv_loss(self, m, DZ):
        """
        Db2{n} = DZ2 * dZ2/db2 = DZ2{n} * 1{n} = dZ2{n}
        The derivative of the loss with respect to particular values of b2
        To bring m back in the picture, we have to average over all of the losses accrued
        during the training run. Namely m training examples:
        (16)    Db2{n} = 1/m * DZ2 * dZ2/dw2 = 1/m * 1{n} dot DZ2{n,m} = 1/m * np.sum(DZ2{n,m})
        """
        return 1 / m * np.sum(DZ, axis=1, keepdims=True)


class Transformation:
    """
    Transformer between output of previous layer (A) to input of next layer (Z)

    """

    def __init__(self, prev_size, next_step_size):
        self.weights = Weights(prev_size=prev_size, next_size=next_step_size)
        self.biases = Biases(next_size=next_step_size)

    def apply(self, A):
        """
        applies weights and biases to A to return Z
        Z = W dot A + b
        """
        return np.dot(self.weights.W, A) + self.biases.b

    def deriv(self):
        """
        finding dZ2/dA1:
        from (9)     Z2{n,m} = W2{n,n} dot A1{n,m} + b2{n,1}
        let i,j in range(n), dropping m for now
        Z2[i] = W2[i]{n} dot A1{n} + b2[i] = {sum over j} W2[i][j] * A1[j] + b2[i]
        dZ2[i]/dA1[j] = W2[j,i]
        dZ2/dA1[j] = W2[j]
        dZ2/dA1 = W2
        adding m back in: let k,l in range(m):
        dZ2[:,k]/dA1[:,l] = 0 if l!= k, else W2
        dZ2[:,k]/dA1[:,k] = W2
        dZ2/dA1{n,n} = W2{n,n}
        """
        return self.weights.W

    def deriv_loss(self, DZ):
        """
        Dropping m again for a moment:
        DA1{n} = DZ2 * dZ2/dA1 = DZ2{n} * W2{n,n} = W2.T{n,n} dot DZ2{n}
        The derivative of loss with respect to a particular A1 value
        Bringing m back in the picture is easy:
        {17}    DA1{n,m} = W2.T{n,n} dot DZ2{n,m}

        DZ = self.next_step.deriv_loss(self.apply(A))
        """
        return np.dot(self.weights.W.T, DZ)
