import matplotlib.pyplot as plt
import numpy as np
from typing import List

from learnt.regression import polynomial_dataframe, get_residual_sum_squares


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

