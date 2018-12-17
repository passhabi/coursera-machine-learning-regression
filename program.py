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

# 9. Consider a simple model with 2 features: ‘sqft_living’ and ‘bedrooms’. The output is ‘price’.
# First, run get_numpy_data() (or equivalent) to obtain a feature matrix with 3 columns (constant column added).
# Use the entire ‘sales’ dataset for now.
# Normalize columns of the feature matrix. Save the norms of original features as ‘norms’.
# Set initial weights to [1,4,1].
# Make predictions with feature matrix and initial weights.
# Compute values of ro[i]
features_list = ['sqft_living', 'bedrooms']
feature_matrix, output = lr.get_numpy_data(df, features_list, 'price')
feature_matrix, norms = lr.normalize_features(feature_matrix)

# assign a random set of initial weights to inspect the values of ro[i]:
weights = np.array([1., 4., 1.])
pred = lr.predict_outcome(feature_matrix, weights)

# ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ]
ro = np.zeros(len(weights))
for j in range(len(weights)):
    temp = feature_matrix[:, j] * (output - pred + (weights[j] * feature_matrix[:, j]))
    ro[j] = temp.sum()
print(ro)

# 10. Quiz Question: Recall that, whenever ro[i] falls between -l1_penalty/2 and l1_penalty/2,
# the corresponding weight w[i] is sent to zero. Now suppose we were to take one step of coordinate descent
# on either feature 1 or feature 2. What range of values of l1_penalty would not set w[1] zero,
# but would set w[2] to zero, if we were to take a step in that coordinate?
print('of range of values', round(2 * ro[2]), ' up to ', round(2 * ro[1]),
      'l1_penalty would not set w[1] zero, but would set w[2] to zero')
# 161933396 up to 175878940

# 11. Quiz Question: What range of values of l1_penalty would set both w[1] and w[2] to zero,
# if we were to take a step in that coordinate?
print('grater than', round(2 * ro[1]), 'l1_penalty would set both w[1] and w[2] to zero')

# 14. Let us now go back to the simple model with 2 features:
# ‘sqft_living’ and ‘bedrooms’. Using ‘get_numpy_data’ (or equivalent), extract the feature matrix and
#   the output array from from the house data frame.
# Then normalize the feature matrix using ‘normalized_features()’ function.
# Using the following parameters, learn the weights on the sales dataset.
#
# Initial weights = all zeros
# L1 penalty = 1e7
# Tolerance = 1.0

# notice features are normalized
simple_weights = lr.lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights=[0., 0., 0.],
                                                      l1_penalty=1e7, tolerance=1.0)
pred = lr.predict_outcome(feature_matrix, simple_weights)
rss_simple_model = lr.get_residual_sum_squares(output, pred)

# 15. Quiz Question: What is the RSS of the learned model on the normalized dataset?
print('RSS with l1_penalty 1e7 and features, sqft_living and bedrooms is :', rss_simple_model)
# 1.63049248148e+15

# 16. Quiz Question: Which features had weight zero at convergence?
# bedrooms

# Let us split the sales dataset into training and test sets.
# If you are using GraphLab Create, call ‘random_split’ with .8
# ratio and seed=0. Otherwise, please down the corresponding csv files from the downloads section.
train = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
test = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)

# 18. Create a normalized feature matrix from the TRAINING data with the following set of features.
# bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_above, sqft_basement,
# yr_built, yr_renovated

features_list = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
                 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']

feature_matrix, output = lr.get_numpy_data(train, features_list, 'price')
features_matrix, norms = lr.normalize_features(feature_matrix)

# 19. First, learn the weights with l1_penalty=1e7, on the training data. Initialize weights to all zeros,
# and set the tolerance=1. Call resulting weights’ weights1e7’, you will need them later.

initial_weights = np.zeros(len(features_list) + 1)
weights1e7 = lr.lasso_cyclical_coordinate_descent(features_matrix, output, initial_weights, l1_penalty=1e7,
                                                  tolerance=1.)

# 20. Quiz Question: What features had non-zero weight in this case?
print(lr.get_included_features_in_model(features_list, weights1e7).keys())
# constant :  24429600.234403126
# sqft_living :  48389174.77154895
# waterfront :  3317511.2149216533
# view :  7329961.811714256

# 21. Next, learn the weights with l1_penalty=1e8, on the training data. Initialize weights to all zeros,
# and set the tolerance=1. Call resulting weights ‘weights1e8’, you will need them later.

weights1e8 = lr.lasso_cyclical_coordinate_descent(features_matrix, output, initial_weights, l1_penalty=1e8,
                                                  tolerance=1.)

# 22. Quiz Question: What features had non-zero weight in this case?
print(lr.get_included_features_in_model(features_list, weights1e8).keys())
# constant 71114625.7528


# 23. Finally, learn the weights with l1_penalty=1e4, on the training data.
# Initialize weights to all zeros, and set the tolerance=5e5. Call resulting weights ‘weights1e4’,
# you will need them later. (This case will take quite a bit longer to converge than the others above.)

weights1e4 = lr.lasso_cyclical_coordinate_descent(features_matrix, output, initial_weights, l1_penalty=1e4,
                                                  tolerance=5e5)
# 24. Quiz Question: What features had non-zero weight in this case?
print(lr.get_included_features_in_model(features_list, weights1e4))

# 25. Recall that we normalized our feature matrix, before learning the weights.
# To use these weights on a test set, we must normalize the test data in the same way.
# Alternatively, we can rescale the learned weights to include the normalization,
# so we never have to worry about normalizing the test data:
# features, norms = lr.normalize_features(feature_matrix)
# weights_normalized = weights / norms

# 26. Create a normalized version of each of the weights learned above.
# (‘weights1e4’, ‘weights1e7’, ‘weights1e8’). To check your results, if you call ‘normalized_weights1e7’
# the normalized version of ‘weights1e7’, then
# print normalized_weights1e7[3]
# should print 161.31745624837794.

normalized_weights1e4 = weights1e4 / norms
normalized_weights1e7 = weights1e7 / norms
normalized_weights1e8 = weights1e8 / norms
print(normalized_weights1e7[3])

# 27. Let's now evaluate the three models on the test data.
# Extract the feature matrix and output array from the TEST set.
# But this time, do NOT normalize the feature matrix. Instead, use the normalized version
# of weights to make predictions.

test_feature_matrix, test_output = lr.get_numpy_data(test, features_list, 'price')
pred1e4 = lr.predict_outcome(test_feature_matrix, normalized_weights1e4)
pred1e7 = lr.predict_outcome(test_feature_matrix, normalized_weights1e7)
pred1e8 = lr.predict_outcome(test_feature_matrix, normalized_weights1e8)

RSS1e4 = lr.get_residual_sum_squares(test_output, pred1e4)
RSS1e7 = lr.get_residual_sum_squares(test_output, pred1e7)
RSS1e8 = lr.get_residual_sum_squares(test_output, pred1e8)

# 28. Quiz Question: Which model performed best on the test data?
# model with 1e7 l1_penalty
