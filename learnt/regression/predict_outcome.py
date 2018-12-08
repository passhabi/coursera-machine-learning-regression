import numpy as np


def predict_outcome(feature_matrix, weights):
    """

    :param feature_matrix: N×D matrix
    :param weights: Estimated weights by gradient decent algorithm. its a vector 1×D
    :return: predicted output base on the given weights. It is a N×1 vector
    """
    weights = weights.reshape(-1, 1)
    predictions = np.dot(feature_matrix, weights)
    return predictions

