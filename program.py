import pandas as pd
from learnt import regression as lr
import matplotlib.pyplot as plt
import numpy as np

# import data:
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int,
              'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': float, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

df = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
train = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
test = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)

# 11. Convert the training set and test set using the ‘get_numpy_data’ function.e.g. in Python:
simple_feature_matrix, output = lr.get_numpy_data(train, features=['sqft_living'], output='price')
simple_test_feature_matrix, test_output = lr.get_numpy_data(test, features=['sqft_living'], output='price')

# 12. First, let’s consider no regularization.
#   Set the L2 penalty to 0.0 and run your ridge regression algorithm to learn the weights of the simple model.
step_size = 1e-12
initial_weights = [0., 0.]
simple_weights_0_penalty = lr.ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights,
                                                                step_size, l2_penalty=0.0, max_iterations=1000)
# 13. Next, let’s consider high regularization.
#   Set the L2 penalty to 1e11 and run your ridge regression to learn the weights of the simple model.
#   Use the same parameters as above.
simple_weights_high_penalty = lr.ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights,
                                                                   step_size, l2_penalty=1e11, max_iterations=1000)

# 14. the following piece of code will plot the two learned models. (The blue line is for the model with
#   no regularization and the red line is for the one with high regularization.)
plt.plot(simple_feature_matrix[:, 1], output, 'k.',  # simple_feature_matrix[:, :] contains 1 constant and sqft_living
         # which we don't care about constant at plotting.
         simple_feature_matrix[:, 1], lr.predict_outcome(simple_feature_matrix, simple_weights_0_penalty), 'b-',
         simple_feature_matrix[:, 1], lr.predict_outcome(simple_feature_matrix, simple_weights_high_penalty), 'r-')
plt.show()
