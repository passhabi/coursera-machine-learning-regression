import numpy as np


def predict_outcome(feature_matrix, weights):
    weights = weights.reshape(-1, 1)
    predictions = np.dot(feature_matrix, weights)
    return predictions

