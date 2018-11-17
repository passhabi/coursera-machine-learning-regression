import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# import data:
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int,
              'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': float, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

train_valid_shuffled = pd.read_csv('wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)

' <-------------------------- Start of Regression functions --------------------------> '


def get_numpy_data(data_frame, features_str, output_str):
    """
    dataframe preparation
    :param data_frame:
    :param features_str:
    :param output_str:
    :return: feature_matrix, weights
    """
    data_frame = data_frame.copy()
    data_frame['constant'] = 1  # add a constant column to an SFrame
    # prepend variable 'constant' to the features list
    features_str = ['constant'] + features_str
    # select the columns of data_frame given by the ‘features’ list into the Frame ‘features_frame’
    features_matrix = data_frame[features_str]
    # this will convert the features_frame into a numpy matrix
    features_matrix = np.array(features_matrix)
    # assign the column of data_frame associated with the target to the variable ‘output_array’
    output_array = data_frame[output_str]
    # this will convert the Array into a numpy array:
    output_array = np.array(output_array).reshape(-1, 1)
    return features_matrix, output_array


def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return predictions


def feature_derivative(errors, feature):
    # -2H^T(y-HW)   =>  -2H^T(error)    =>  2 * H^T * error
    feature = 2 * feature
    derivative = np.dot(feature, errors)
    return derivative


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


def get_residual_sum_squares(output, predicted_output):
    """
    :param output: price or Y
    :param predicted_output: predicted y or Y_hat
    :return:
    """
    residual = output - predicted_output
    rss = np.dot(residual.T, residual)
    return rss


def polynomial_dataframe(feature: pd.DataFrame.axes, degree: int):
    """
    Generate polynomial features up to degree 'degree'.


    :param feature: Is pandas.Series, float, double type
    :param degree: to power
    :return: data frame to the degree power
    """
    # assume that degree >= 1
    # initialize the data frame:

    poly_df = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_df['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree + 1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_df[name] = feature ** power

    return poly_df


def fit_poly_model(order, train_data, feature: str, valid_data=None, output: str = 'price',
                   l2_penalty=1e-9,
                   normalization: bool = True, model_plot: bool = False, color_scheme: List[str] = None,
                   pause_plotting_time=5):
    """
    It makes a polynomial dataframe by feature to the power of 'order' and plots the feature as x and cost as y, It
    fits the model to the polynomial dataframe using sikit-learn.\n

    :param order:
    :param train_data:
    :param feature:
    :param valid_data:
    :param output:
    :param l2_penalty:
    :param normalization:
    :param model_plot:
    :param color_scheme: a list of color, first entry for scatter points and second for plotting. e.g. ['aqua', 'blue']
        or ['red', 'crimson']
    :param pause_plotting_time:
    :return:
    """
    # an 'order' degree polynomial :
    poly_data = polynomial_dataframe(train_data[feature], order)
    poly_data[output] = train_data[output]

    # compute the regression weights for predicting sales[‘price’]
    #   based on the 1 degree polynomial feature ‘sqft_living’:
    from sklearn.linear_model import Ridge
    # make a new instance of the object:
    model = Ridge(alpha=l2_penalty, normalize=normalization)
    #   convert data frame to numpy array to prevent shape error with sikit-learn:
    x = np.array(poly_data.iloc[:, :-1])
    y = np.array(poly_data[output]).reshape(-1, 1)

    model.fit(x, y)

    # store all coefficient in poly1_weights array:
    poly_weights = model.intercept_
    for i in range(0, len(model.coef_)):
        poly_weights = np.append(poly_weights, model.coef_[i])

    # Plotting the model, features Xs vs observation Y:
    if model_plot:
        # produce a scatter plot of the training data (just square feet vs price) with fitted model:
        if color_scheme is not None:
            # plot without default color:
            plt.scatter(poly_data['power_1'], poly_data[output], c=color_scheme[0])
            plt.plot(x[:, 0], model.predict(x), c=color_scheme[1])
        else:
            # plot with default color but in different figures:
            import random
            num_figure = random.randint(0, 1000)
            plt.figure(num_figure)
            plt.scatter(poly_data['power_1'], poly_data[output])
            plt.plot(x[:, 0], model.predict(x), c='red')
            plt.figure(num_figure).show()
        plt.pause(pause_plotting_time)

    # compute rss:
    train_rss = get_residual_sum_squares(y, model.predict(x))
    # compute rss on validation set:
    if valid_data is None:
        # Then we don't need validation_rss:
        validation_rss = None
    else:
        poly_data_valid = polynomial_dataframe(valid_data[feature], order)
        poly_data_valid[output] = valid_data[output]

        x_valid = np.array(poly_data_valid.iloc[:, :-1])
        y_valid = np.array(poly_data_valid[output]).reshape(-1, 1)
        # get ready validation rss to return:
        validation_rss = get_residual_sum_squares(y_valid, model.predict(x_valid))

    return poly_weights, train_rss, validation_rss


def k_fold_cross_validation(k: int, l2_penalty: float, data: pd.DataFrame, output: pd.DataFrame):
    from sklearn.linear_model import Ridge

    n = len(output)  # number of data points (observations)
    valid_error = 0
    for i in range(k):
        start = int((n * i) / k)  # start index
        end = int((n * (i + 1)) / k)  # end index

        # split data to valid set and train set:
        x_valid_set = data[start:end + 1]  # omitted segment for testing training set
        #   same for the output array:
        y_valid_set = output[start:end + 1]

        #   assume valid_set with - and train segments with + and [] for selecting area.
        x_train_set = data[0:start]  # [+++]---++++++++++
        x_train_set = x_train_set.append(data[end + 1:])  # [+++]---[++++++++++]
        #   same again for output:
        y_train_set = output[0:start]
        y_train_set = y_train_set.append(output[end + 1:])

        # convert to numpy array:
        features_train = x_train_set.iloc[:, :].values
        output_train = y_train_set.iloc[:].values
        #   do the same for validation set
        features_valid = x_valid_set.iloc[:, :].values
        output_valid = y_valid_set.iloc[:].values

        # TODO: use my future ridge regression method instead of skiti-learn
        model = Ridge(l2_penalty, normalize=True)
        model.fit(features_train, output_train)

        # compute RSS for validation set:
        prediction = model.predict(features_valid)
        # store currant RSS:
        valid_error += get_residual_sum_squares(output_valid, prediction)

    return valid_error / k


' <--------------------------- End of Regression functions ---------------------------> '

# 16.Once we have a function to compute the average validation error for a model,
#   we can write a loop to find the model that minimizes the average validation error:
poly15_sqft_living = polynomial_dataframe(train_valid_shuffled['sqft_living'], 15)
validation_errors_dict = {}
for l2_penalty in np.logspace(3, 9, 13):
    error = k_fold_cross_validation(5, l2_penalty, poly15_sqft_living, train_valid_shuffled['price'])
    validation_errors_dict[l2_penalty] = error
# print dict:
validation_errors_dict

# Review: seems I get a logical error, check the k_fold_cross_validation and choosing l2_penalty methods, later.

# 17. Quiz Question:
#   What is the best value for the L2 penalty according to 10-fold validation?
# Answer:
#   1000

# 18. Once you found the best value for the L2 penalty using cross-validation,
#   it is important to retrain a final model on all of the training data using this value of l2_penalty.
#  This way, your final model will be trained on the entire dataset:

from sklearn.linear_model import Ridge

model = Ridge(alpha=min(validation_errors_dict.keys()))
# prepare data to fit a model:
features_list = list(poly15_sqft_living.columns.values)
data_frame = poly15_sqft_living
data_frame['price'] = train_valid_shuffled['price']
feature_matrix, output_array = get_numpy_data(data_frame, features_list, 'price')
model = model.fit(feature_matrix, output_array)

# lets predict for test data to see the generalization error:
poly15_test = polynomial_dataframe(test['sqft_living'], 15)
poly15_test['price'] = test['price']
x, y = get_numpy_data(poly15_test, features_list, 'price')
# and now prediction for test:
prediction = model.predict(x)
# compute RSS on test data:
RSS = get_residual_sum_squares(y, prediction)
print(RSS)

# 19. Quiz Question: Using the best L2 penalty found above, train a model using all training data.
#   What is the RSS on the TEST data of the model you learn with this L2 penalty?
# Answer:
#   1.06586933e+18
