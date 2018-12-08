import numpy as np


def get_residual_sum_squares(output, predicted_output):
    """
    :param output: vector
    :param predicted_output: predicted y vector
    :return:
    """
    residual = output - predicted_output
    rss = np.dot(residual.T, residual)
    return rss[0][0]    # remove the [[brackets]] at the return value
