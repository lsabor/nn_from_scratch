"""utility functions"""


def ReLU(n: float) -> float:
    """rectified linear unit activation function"""
    return n if n > 0 else 0
