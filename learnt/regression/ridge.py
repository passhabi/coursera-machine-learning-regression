import numpy as np
from typing import List
from learnt.regression import predict_outcome


def feature_derivative_ridge(errors, feature, weight, l2_penalty: float, feature_is_constant: bool):
    """
    Computing the derivative of the regression cost function.
    Recall that the cost function is the sum over the data points of the squared difference between an observed output
    and a predicted output, plus the L2 penalty term.

    Parameters:
    ----------
    :param errors: ndarray
    :param feature: column of a feature.
    :param feature_is_constant: Set true when given column of a feature is a constant.
    :param weight: ndarray
    :param l2_penalty: (lambda) Regularization tuning parameter
    :return: derivation (ndarray)
    """
    # (y-HW)ᵀ(y-HW) + λ |W|²    is our cost function. to derive this; we'll get following:
    # -2Hᵀ(y-HW) + 2λW

    # IMPORTANT: We will not regularize the constant. Thus, in the case of the constant,
    #   the derivative is just twice the sum of the errors (without the 2λw[0] term).
    # If feature_is_constant is True, derivative is twice the dot product of errors and feature
    errors = np.reshape(errors, [-1, 1])  # need error to be a n×1 vector
    derivative = 2 * np.dot(feature, errors)  # 1×n dot product n×1 gives us a scalar
    # simple form of code above:
    # derivative = feature * errors
    # derivative = 2 * sum(derivative)
    if not feature_is_constant:
        # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
        derivative = derivative + 2 * (l2_penalty * weight)
    # Noticed omitted -1?! We are adding it at the updating weights term (at ridge gradient decent function).
    return derivative


'''
# To test your feature derivative function, run the following:

import pandas as pd
from learnt.regression import get_numpy_data
from learnt.regression import predict_outcome

dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int,
              'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': float, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

df = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
example_features, example_output = get_numpy_data(df, ['sqft_living'], 'price')
my_weights = np.array([1., 10.])
test_predictions = predict_outcome(example_features, my_weights)
errors = test_predictions - example_output  # prediction errors

# next two lines should print the same values
print(feature_derivative_ridge(errors, example_features[:, 1], my_weights[1], 1, False))
print(np.sum(errors * example_features[:, 1]) * 2 + 20.)
print('')

# next two lines should print the same values
print(feature_derivative_ridge(errors, example_features[:, 0], my_weights[0], 1, True))
print(np.sum(errors) * 2.)
'''


def ridge_regression_gradient_descent(feature_matrix, output, initial_weights: List[float], step_size,
                                      l2_penalty: float,
                                      max_iterations: int = 100):
    if type(initial_weights[0]) != float:
        # make sure auto casting, (float to int) doesn't happen at updating weights[i].
        raise Exception('initial_weights setted with an int number instead of a float')

    weights = np.array(initial_weights)  # make sure it's a numpy array
    while 0 < max_iterations:  # while not reached maximum number of iterations:
        # compute the predictions using your predict_output() function
        predictions = predict_outcome(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = predictions - output  # predictions is n×1 so we need output to be n×1 too

        for i in range(len(weights)):  # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            is_constant: bool = False
            if i == 0:  # when i is equal to 0, you are computing the derivative of the constant!
                is_constant = True

            # compute the derivative for weight[i]:
            derivative = feature_derivative_ridge(errors, feature_matrix[:, i], weights[i], l2_penalty, is_constant)
            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - step_size * derivative
        max_iterations -= 1
    return weights.reshape(-1, 1)
