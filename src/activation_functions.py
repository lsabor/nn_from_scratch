"""
Activation Functions for neural net
"""

import numpy as np

from loss_functions import CrossEntropy


class ActivationLayer:
    """
    Represents an activation function
    Arguments:
    - size = number of neurons in layer
    - next = next step in the neural net
    Notation: input is Z, output is A
    """

    name = ...
    equation = ...

    def __init__(self, size, next_step):
        self.size = size
        self.next = next_step

    def deriv(self, Z):
        ...

    def deriv_loss(self, DA, dZ):
        """
        returns the derivative matrix of Loss with respect to input to ReLU Z
        - override this if there is a mathematical shortcut or is not caluclated element-wise
        - is element-wise multiplication by default because apply() is usually element-wise
        DA = self.next.deriv_loss(self.apply(Z))
        dZ = self.deriv(Z)
        """
        print(f"{self.name} - deriv_loss")
        print(f"{dZ = }")
        return DA * dZ


class Softmax(ActivationLayer):
    """
    softmax activation function
    Softmax(Z)[i] = e^Z[i] / sum_i(e^Z)
    Y_hat = Softmax(Z)
    """

    name = "Softmax"
    equation = "e^Z[i] / sum_i(e^Z)"

    def apply(self, Z: np.array):
        """
        returns the softmax array of Z
        called Y_hat since only used at termination of nerual net
        """
        # TODO: reapply stabilizing
        # collapses 1 dim of array
        max_Z = np.amax(Z, axis=0).reshape(1, Z.shape[1])  # Get the column-wise maximum
        eZ = np.exp(Z - max_Z)  # For stability
        return eZ / eZ.sum(axis=0, keepdims=True)

    def deriv(self, Y_hat):
        """
        returns a derivative matrix of how each value in Z effects
        each value in A
        A = self.apply(Z)
        """
        """
        dY_hat/dZ2:  
        dY_hat/dZ2.shape should be {n,m}  
        from (10)    Y_hat{n,m} = the estimate of Y = softmax(Z2{n,m}):  
        given some i,j in range(n) and k,l in range(m):  
        Y_hat[i,k] changes with respect to Z2[j,l] only when k == l  
        for simplicity, assume k=l and thus drop those terms  
        dY_hat[i]/dZ2 has dimension {n}  
        dY_hat[i]/dZ2[j] =   
            if i == j --> softmax(Z2[j])*(1-softmax(Z2[j])  
            if i != j --> -softmax(Z2[i])*softmax(Z2[j])  
        dY_hat/dZ2 has dimension [n,n] for each entry in m  
        dY_hat/dZ2[i,j,k] =  
            if i == j --> softmax(Z2[j,k])*(1-softmax(Z2[j,k])  
            if i != j --> -softmax(Z2[i,k])*softmax(Z2[j,k])  
        for simplicity, call p[i, ...] = softmax(Z2[i, ...]). Thus:  
        (13)    dY_hat/dZ2[i,j,k]{n,n,m} =  
                if i == j --> p[j,k]*(1-p[j,k])  
                if i != j --> -p[i,k]*p[j,k] 
        """
        softmax = Y_hat
        identity = np.eye(softmax.shape[-1])
        # must instantiate with proper dims first
        t1 = np.zeros(softmax.shape + (softmax.shape[-1],), dtype=np.float32)
        t2 = np.zeros(softmax.shape + (softmax.shape[-1],), dtype=np.float32)
        t1 = np.einsum("ij,ik->ijk", softmax, softmax)  # handles rest when i != j
        t2 = np.einsum("ij,jk->ijk", softmax, identity)  # handles only when i == j
        return t2 - t1

    def deriv_loss(self, Y_hat):
        """
        returns the derivative matrix of Loss with respect to input to softmax Z
        - currently only supports loss == CrossEntropy
        Y_hat = self.apply(Z)
        """
        if isinstance(self.next, CrossEntropy):
            """
            DZ2 = dL/dZ2:
            DZ2.shape should be {n,m}
            DZ2 = dL/dY_hat * dY_hat/dZ2
            for now, drop m, so L has dim 1 while Z2 has dim {n}
            let i,j in range(n)
            from (13)   dY_hat/dZ2[i,j,k]{n,n,m} =
                            if i == j --> p[j,k]*(1-p[j,k])
                            if i != j --> -p[i,k]*p[j,k]:
            dL/dZ2[j] = sum over i of dL/dY_hat[i] * dY_hat[i]/dZ2[j]
                = {when i == j} - Y[j]/Y_hat[j] * Y_hat[j]*(1-Y_hat[j])
                + {sum over i when i != j of} (- (Y[i] / Y_hat[i]) * -Y_hat[i]*Y_hat[j] )
                = -Y[j] * (1 - Y_hat[j]) - Y_hat[j] * {sum over i when i != j of} Y[i]
                = -Y[j] + Y[j] * Y_hat[j] - Y_hat[j] * (-Y[j] + {sum over i of} Y[i]) # added Y[j] into summation
                = -Y[j] + Y_hat[j] * (-Y[j] - (-Y[j] + 1)) # NOTE: {sum over i of} Y[i] = 1 since
                                                            # Y[i] = 0 for all but 1 i, where it equals 1
                = -Y[j] + Y_hat[j] * 1 = -Y[j] + Y_hat[j]
            Adding back in k in range(m):
            dL/dZ2[j,k] = -Y[j,k] + Y_hat[j,k]
            (14)    DZ2{n,m} = -Y + Y_hat
            """
            print(f"{self.name} - deriv_loss")
            return -self.next.Y + Y_hat

        raise NotImplementedError(
            "currently only implemented as last layer activation "
            + "function with CrossEntropy as loss function"
        )


class ReLU(ActivationLayer):
    """
    Rectified Linear Unit activation function
    ReLU(Z)[i] = max(Z[i], 0)
    """

    name = "Rectified Linear Unit"
    equation = "max(Z[i], 0)"

    def apply(self, Z: np.array):
        """rectified linear unit activation function"""
        return np.maximum(Z, 0)

    def deriv(self, Z):
        """
        returns a derivative matrix of how each value in Z effects
        each value in A
        """
        """
        dA/dZ[i] = 1 if Z[i] > 1, else 0
        """
        return Z > 0
