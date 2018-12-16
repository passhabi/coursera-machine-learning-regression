from learnt.regression import predict_outcome


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    """ coordinate descent that minimizes the cost function over a single feature i.

    The function accept a feature matrix, an output, current weights, l1 penalty,
    and index of feature to optimize over. The function return new weight for feature i.
    Note that the intercept (weight 0) is not regularized.
    :param i: index
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
    ro_i = feature_matrix[:, i] * (output - prediction + feature_matrix[:, i] * weights[i])
    ro_i = ro_i.sum()
    if i == 0:  # intercept -- do not regularize
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
