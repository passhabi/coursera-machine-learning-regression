import pandas as pd
from learnt import regression as lr
import matplotlib.pyplot as plt
import numpy as np

# Import data:
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int,
              'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': float, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

df = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
train_df = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
test_df = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)

# 11. Convert the training set and test set using the ‘get_numpy_data’ function.e.g. in Python:
simple_feature_matrix, output = lr.get_numpy_data(train_df, features=['sqft_living'], output='price')
simple_test_feature_matrix, test_output = lr.get_numpy_data(test_df, features=['sqft_living'], output='price')

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
         simple_feature_matrix[:, 1],
         lr.predict_outcome(simple_feature_matrix, simple_weights_0_penalty).reshape(-1, 1), 'b-',
         simple_feature_matrix[:, 1],
         lr.predict_outcome(simple_feature_matrix, simple_weights_high_penalty).reshape(-1, 1), 'r-')
plt.show()

# 15. Quiz Question: What is the value of the coefficient for sqft_living that you learned with no regularization,
#   rounded to 1 decimal place? What about the one with high regularization?
print("Coefficient of sqft_living with no regularization and a simple feature:",
      round(simple_weights_0_penalty[1][0], 1))  # [1][0] is sqft_living

# 16. Quiz Question: Comparing the lines you fit with the with no regularization versus high regularization,
#   which one is steeper?
# The model with no regularization penalty is steeper (blue line).

# 17. Compute the RSS on the TEST data for the following three sets of weights:
#
# The initial weights (all zeros)
# The weights learned with no regularization
# The weights learned with high regularization

# Get predictions associated with each weights on test data frame.
prediction_with_initial_weights = lr.predict_outcome(simple_test_feature_matrix, initial_weights)
prediction_with_no_penalty = lr.predict_outcome(simple_test_feature_matrix, simple_weights_0_penalty)
prediction_with_high_penalty = lr.predict_outcome(simple_test_feature_matrix, simple_weights_high_penalty)

# Compute RSS for each one:
rss_dict = {"initial_weights": lr.get_residual_sum_squares(test_output, prediction_with_initial_weights),
            "no_penalty": lr.get_residual_sum_squares(test_output, prediction_with_no_penalty),
            "high_penalty": lr.get_residual_sum_squares(test_output, prediction_with_high_penalty)}

# 18. Quiz Question: What are the RSS on the test data for each of the set of weights above
#   (initial, no regularization, high regularization)?

# Print them out:
print("Residual some of squares with initial weights: ", round(rss_dict["initial_weights"]))
print("Residual some of squares with no penalty: ", round(rss_dict["no_penalty"]))
print("Residual some of squares with high penalty: ", round(rss_dict["high_penalty"]))

# 19. Let us now consider a model with 2 features: [ ‘sqft_living’, ‘sqft_living_15’].
#   First, create Numpy version of your training and test data with the two features.
model_features = ['sqft_living', 'sqft_living15']
feature_matrix, output = lr.get_numpy_data(train_df, model_features, 'price')
test_feature_matrix, test_output = lr.get_numpy_data(test_df, model_features, 'price')

# 20.First, let’s consider no regularization.
#   Set the L2 penalty to 0.0 and run your ridge regression algorithm. Use the following parameters:
initial_weights = [0., 0., 0.]
step_size = 1e-12
max_iterations = 1000

multiple_weights_0_penalty = lr.ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size
                                                                  , 0.0, max_iterations)

# 21. Next, let’s consider high regularization. Set the L2 penalty to 1e11.and run your ridge regression to learn
#   the weights of the simple model. Use the same parameters as above:
multiple_weights_high_penalty = lr.ridge_regression_gradient_descent(feature_matrix, output, initial_weights
                                                                     , step_size, 1e11, max_iterations)

# 22. Quiz Question: What is the value of the coefficient for ‘sqft_living’ that you learned with no regularization,
#   rounded to 1 decimal place? What about the one with high regularization?
print("Coefficient of sqft_living with multiple feature and no regularization:",
      round(multiple_weights_0_penalty[1][0], 1))  # [1][0] is sqft_living

print("Coefficient of sqft_living with multiple feature and high regularization:",
      round(multiple_weights_high_penalty[1][0], 1))  # [1][0] is sqft_living

# 23. Compute the RSS on the TEST data for the following three sets of weights:
#
# The initial weights (all zeros)
# The weights learned with no regularization
# The weights learned with high regularization

# Get predictions associated with each weights on test data frame:
prediction_with_initial_weights = lr.predict_outcome(test_feature_matrix, initial_weights)
prediction_with_no_penalty = lr.predict_outcome(test_feature_matrix, multiple_weights_0_penalty)
prediction_with_high_penalty = lr.predict_outcome(test_feature_matrix, multiple_weights_high_penalty)

# Compute RSS for each one this time for a model with multiple feature:
rss_dict = {"initial_weights": lr.get_residual_sum_squares(test_output, prediction_with_initial_weights),
            "no_penalty": lr.get_residual_sum_squares(test_output, prediction_with_no_penalty),
            "high_penalty": lr.get_residual_sum_squares(test_output, prediction_with_high_penalty)}

# finally print them out again:
print("RSS with initial weights on multiple feature: ", round(rss_dict["initial_weights"]))
print("RSS with no penalty on multiple feature: ", round(rss_dict["no_penalty"]))
print("RSS with high penalty on multiple feature: ", round(rss_dict["high_penalty"]))

# 25. Predict the house price for
#   the 1st house in the test set using the no regularization and high regularization models.
print("Actual price of 1st house:", output[0])
print("House price for 1st house on test with no regularization:", prediction_with_no_penalty[0][0])
# cause its a nested array, need to add another [0]
print("House price for 1st house on test with high regularization:", prediction_with_high_penalty[0][0])

# 26. Quiz Question: What's the error in predicting the price of the first house in the test set
#   using the weights learned with no regularization? What about with high regularization?
print("(no_penalty) error for 1st house:", prediction_with_no_penalty[0][0] - test_output[0])
print("(high_penalty) error for 1st house:", prediction_with_high_penalty[0][0] - test_output[0])
