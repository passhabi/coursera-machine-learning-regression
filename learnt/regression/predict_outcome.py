import numpy as np


def predict_outcome(feature_matrix, weights):
    """

    :param feature_matrix: N×D
    :param weights: Estimated weights by gradient decent algorithm. its a vector 1×D
    :return: (ndarray) predicted output base on the given weights. It is a 1×N vector
    """
    # make sure it's a numpy array:
    if type(weights) == list:
        weights = np.array(weights)
    if type(feature_matrix) == list:
        feature_matrix = np.array(feature_matrix)

    weights = weights.reshape(-1, 1)
    predictions = np.dot(feature_matrix, weights)
    return predictions.reshape(1, -1)  # reshape vector to 1×N (default shape).
