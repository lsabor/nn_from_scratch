import os
from typing import Tuple
import pandas as pd
import numpy as np


def get_data_from_csv(file: str) -> Tuple[np.array, int, int]:
    """takes data from file (csv type) and returns
    a shuffled version of the data in an np.array form,
    along with two ints:
    m - number of test examples
    n - number of points per example (including integrated labels)"""
    assert os.path.exists(file), f"{file} does not exist"

    data = pd.read_csv(file)
    m, n = data.shape
    data = np.array(data)
    np.random.shuffle(data)

    return (data, m, n)


def get_labels_and_data_1st_column(data: np.array) -> Tuple[np.array, np.array]:
    """takes an np.array of data, returns (Transposed) labels (Y) and data (X)"""
    data = data.T
    Y = data[0]
    X = data[1:]
    return Y, X
