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


def import_mnist():

    # DATA FROM HERE: https://pjreddie.com/projects/mnist-in-csv/
    file_test = "../data/MNIST/mnist_test.csv"
    file_train = "../data/MNIST/mnist_train.csv"

    data_test, m_test, n_test = get_data_from_csv(file_test)
    Y_test, X_test = get_labels_and_data_1st_column(data_test)

    data_train, m_train, n_train = get_data_from_csv(file_train)
    Y_train, X_train = get_labels_and_data_1st_column(data_train)

    assert n_test == n_train
    n = n_test
    m = m_test + m_train

    """making sure that our Y_test/Y_train are actually labels"""

    assert Y_test.max() == 9
    assert Y_train.max() == 9
    assert X_test[0].max() != 9
    assert X_train[0].max() != 9

    return X_train, Y_train, X_test, Y_test
