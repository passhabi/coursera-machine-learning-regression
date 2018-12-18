import numpy as np


def get_residual_sum_squares(output, predicted_output):
    """
    :param output: vector
    :param predicted_output: predicted y vector
    :return:
    """
    residual = output - predicted_output
    square = residual ** 2
    rss = square.sum()
    return rss
