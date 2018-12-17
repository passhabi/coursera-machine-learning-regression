import numpy as np
from learnt.regression import predict_outcome


def lasso_coordinate_descent_step(index, feature_matrix, output, weights, l1_penalty):
    """ coordinate descent that minimizes the cost function over a single feature i.

    The function accept a feature matrix, an output, current weights, l1 penalty,
    and index of feature to optimize over. The function return new weight for feature i.
    Note that the intercept (weight 0) is not regularized.
    :param index: index
    :param feature_matrix:
    :param output:
    :param weights:
    :param l1_penalty:
    :return: new weight
    """
    # todo: check if features are normalized or not

    # compute prediction
    prediction = predict_outcome(feature_matrix, weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = feature_matrix[:, index] * (output - prediction + feature_matrix[:, index] * weights[index])
    ro_i = ro_i.sum()
    if index == 0:  # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty / 2.:
        new_weight_i = ro_i + l1_penalty / 2.
    elif ro_i > l1_penalty / 2.:
        new_weight_i = ro_i - l1_penalty / 2.
    else:
        new_weight_i = 0.
    return new_weight_i


'''
# If you are using Numpy, test your function with the following snippet:
# should print 0.425558846691
import math
import numpy as np

print(lasso_coordinate_descent_step(1, np.array([[3. / math.sqrt(13), 1. / math.sqrt(10)],
                                                 [2. / math.sqrt(13), 3. / math.sqrt(10)]]), np.array([1., 1.]),
                                    np.array([1., 4.]), 0.1))
'''


def lasso_cyclical_coordinate_descent(feature_matrix: np.ndarray, output: np.ndarray, initial_weights: list,
                                      l1_penalty: float, tolerance: float):
    """ Cyclical coordinate descent.

        Cyclical coordinate descent where we optimize coordinates 0, 1, ..., (d-1) in order and repeat.
    For each iteration:

    loop over features in order and perform coordinate descent, measure
    how much each coordinate changes.
    After the loop, if the maximum change across all coordinates is falls below
    the tolerance, It stops. Otherwise, it backs to the previous step.
    and finally returns the weights.

    :param feature_matrix:
    :param output:
    :param initial_weights:
    :param l1_penalty:
    :param tolerance:
    :return: ndarray obtained weights by l1_penalty
    """
    d = len(initial_weights)  # number of coordinates (W or features)
    weights = np.array(initial_weights)  # make sure its a numpy array
    diff_weights = np.zeros(d)  # store magnitude of each step we take
    converged = False
    while not converged:
        max_step = 0
        for i in range(d):
            current_weight_i = weights[i]
            # update weights for each coordinate:
            weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)

            # check how big the algorithm takes steps towards each coordinates:
            new_weight_i = weights[i]
            diff_weights[i] = abs(new_weight_i - current_weight_i)  # differences for coordinate i_th
            # take the biggest step, cycling through all coordinates:
            max_step = max(diff_weights)

        if max_step < tolerance:
            converged = True
    return weights


def get_included_features_in_model(features_list: list, weights: np.ndarray):
    """ What features are included in your lasso model?

    :param features_list: string list of features name
    :param weights: model weight. intercept must be included
    :return: a dict, a pair of feature name and corresponding weight.
    """
    all_features = ['constant'] + features_list
    included_features = {}
    for i in range(len(weights)):
        if not weights[i] == 0:  # if corresponding weight of features has a value (not zero), then print it.
            # these are the features that our model included:
            # print(all_features[i], ': ', weights[i])
            included_features[all_features[i]] = weights[i]
    return included_features
