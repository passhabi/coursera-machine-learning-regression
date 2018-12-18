import numpy as np


def compute_distances(features_instances, features_query):
    """
    Compute euclidean distance.

        computes the distances from a query house to all training houses.
        The function takes two parameters:
        (i) the matrix of training features and
        (ii) the single feature vector associated with the query.
    """
    #                                         _________________________________________
    # Euclidean distance: distance(xi,xq) = √(xj[1] − xq[1])² + ... + (xj[d] − xq[d])²
    differences = features_instances - features_query  # xj[1] − xq[1]
    square_diff = differences ** 2  # (xj[1] − xq[1])²
    # note: summing up only over features for every squared difference of each observation:
    sum_square_diff = np.sum(square_diff, axis=1)
    distances_matrix = np.sqrt(sum_square_diff)
    return distances_matrix


def k_nearest_neighbors(k, feature_train, features_query):
    """
    preform k-nearest neighbor regression.

    Example:
    -------
        For instance, with 2-nearest neighbor,
        a return value of [5, 10] would indicate that the 6th and 11th training houses are closest to the query house.
    :param k: consider k neighbor
    :param feature_train:
    :param features_query:
    :return:  returns the indices vector of the k closest training houses
    """
    distances_matrix = compute_distances(feature_train, features_query)
    sorted_matrix = np.argsort(distances_matrix)
    return sorted_matrix[:k]  # return k neighbors


def predict_output_of_query(k, features_train, output_train, features_query):
    """
    predict output using k nearest neighbors for given features query

        predicts the value of a given query house. simply,
    takes the average of the prices of the k nearest neighbors in the training set.
    :param k:
    :param features_train:
    :param output_train:
    :param features_query:
    :return:
    """
    knn_indices = k_nearest_neighbors(k, features_train, features_query)
    avg_value = np.mean(output_train[knn_indices])
    return avg_value  # mean of k nearest neighbors is our prediction of given query


def predict_output(k, features_train, output_train, features_query):
    """
    predict outputs using k nearest neighbors for given features query set

        predict the value of each and every house in a query set.
    (The query set can be any subset of the dataset, be it the test set or validation set.)
    The idea is to have a loop where we take each house in the query set as the query house and make a prediction
    for that specific house.

    :param k:
    :param features_train:
    :param output_train:
    :param features_query:
    :return:
    """
    n = len(features_query)  # number of observation we want to predict
    predictions = np.zeros(n)
    for i in range(n):
        # for each feature query that you want to predict its output, do:
        predictions[i] = predict_output_of_query(k, features_train, output_train, features_query[i])

    return predictions
