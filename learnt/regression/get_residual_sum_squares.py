import numpy as np


def get_residual_sum_squares(output, predicted_output):
    """
    :param output: price or Y
    :param predicted_output: predicted y or Y_hat
    :return:
    """
    residual = output - predicted_output
    rss = np.dot(residual.T, residual)
    return rss