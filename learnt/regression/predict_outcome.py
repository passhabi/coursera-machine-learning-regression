import numpy as np


def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return predictions

