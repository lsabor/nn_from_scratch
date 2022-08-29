"""utility functions"""

import numpy as np


def ReLU(n: float) -> float:
    """rectified linear unit activation function"""
    return np.maximum(n, 0)


def ReLU_deriv(n: float) -> int:
    return n > 0
