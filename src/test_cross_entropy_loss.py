import numpy as np


def one_hot_encode(Y: np.array, classes=10):
    # first instantiate 0's which should be an array of len(Y) max(Y)
    one_hot = np.zeros((Y.size, classes))
    one_hot[np.arange(Y.size), Y] = 1
    return one_hot


def cross_entropy_loss(predictions, y_actual):
    num_samples = len(y_actual)
    class_targets = np.array(y_actual)

    y_pred_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
    if len(y_actual.shape) == 1:
        targeted_predictions = y_pred_clipped[[range(num_samples), class_targets]]
    elif len(y_actual.shape) == 2:
        print(f"{y_pred_clipped=}")
        print(f"{y_actual=}")

        targeted_predictions = np.sum(y_pred_clipped * y_actual, axis=1)
        print(f"{targeted_predictions=}")
    negative_log_likelihoods = -np.log(targeted_predictions)


pred = np.array([1, 1, 1])

print(one_hot_encode(pred, 3))
