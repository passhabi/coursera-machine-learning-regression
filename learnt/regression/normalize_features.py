import numpy as np


def normalize_features(feature_matrix: np.ndarray):
    """Normalizes columns of a given feature matrix.

    The function returns a pair ‘(normalized_features, norms)’,
    where the second item contains the norms of original

    :param feature_matrix:
        a matrix of features
    :return:
        normalized features as matrix and its norms
    """

    # normalization formula: hj(x) / √ ∑ hj(x)²
    # simply feature dived by norms
    # the following code using numpy satisfy the norm behavior for each columns:
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_features = feature_matrix / norms
    return normalized_features, norms
