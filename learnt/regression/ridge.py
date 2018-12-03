import numpy as np
from learnt.regression import predict_outcome


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    """
    **Gradient descent Algorithm**


    :param feature_matrix:
    :param output:
    :param initial_weights: 1×n numpy array
    :param step_size:
    :param tolerance:
    :return:
    """
    converged = False
    weights = np.array(initial_weights).reshape(-1, 1)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        predictions = predict_outcome(feature_matrix, weights)
        # compute the errors as predictions - output:
        errors = output - predictions
        gradient_sum_squares = 0  # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors, feature_matrix[:, i])
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares = + derivative ** 2
            # update the weight based on step size and derivative:
            weights[i] = weights[i] + step_size * derivative
        gradient_magnitude = np.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return weights


def feature_derivative_ridge(errors, feature, weight, l2_penalty: float, feature_is_constant: bool):
    """
    Computing the derivative of the regression cost function.
    Recall that the cost function is the sum over the data points of the squared difference between an observed output
    and a predicted output, plus the L2 penalty term.

    Parameters:
    ----------
    :param errors: ndarray
    :param feature: numpy arraylike (matrix)
    :param feature_is_constant: Set true when given feature is a constant.
    :param weight: ndarray
    :param l2_penalty: (lambda) Regularization tuning parameter
    :return: derivation (ndarray)
    """
    # (y-HW)ᵀ(y-HW) + λ |W|²    is our cost function. to derive this; we'll get following:
    # -2Hᵀ(y-HW) + 2λW
    if feature_is_constant:
        # IMPORTANT: We will not regularize the constant. Thus, in the case of the constant,
        #   the derivative is just twice the sum of the errors (without the 2λw[0] term).
        derivative = 2 * errors.sum()
    else:
        derivative = 2 * (np.dot(feature, errors.T) + l2_penalty * weight)
        # Noticed omitted -1?! We are adding it at the updating weights term (at gradient decent function).
    return derivative


'To test your feature derivative function, run the following: '


# from learnt.regression import get_numpy_data
# example_features, example_output = get_numpy_data(data, ['sqft_living'], 'price')
# my_weights = np.array([1., 10.])
# test_predictions = predict_outcome(example_features, my_weights)
# errors = test_predictions - example_output.T  # prediction errors
#
# # next two lines should print the same values
# print(feature_derivative_ridge(errors, example_features[:, 1], my_weights[1], 1, False)[0])
# print(np.sum(errors * example_features[:, 1]) * 2 + 20.)
# print('')
#
# # next two lines should print the same values
# print(feature_derivative_ridge(errors, example_features[:, 0], my_weights[0], 1, True))
# print(np.sum(errors) * 2.)


def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty: float,
                                      max_iterations: int = 100):
    weights = np.array(initial_weights)  # make sure it's a numpy array
    iterate = 0
    while iterate <= max_iterations:  # while not reached maximum number of iterations:
        # compute the predictions using your predict_output() function
        predictions = predict_outcome(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = output - predictions

        for i in range(len(weights)):  # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            # (Remember: when i=0, you are computing the derivative of the constant!)
            is_constant: bool = False
            if i == 0:
                is_constant = True
            derivative = feature_derivative_ridge(errors, feature_matrix[:, i], weights[i], l2_penalty, is_constant)
            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - step_size * derivative
        iterate += 1
    return weights
