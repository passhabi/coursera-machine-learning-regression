import pandas as pd
from learnt import regression as lr
import numpy as np
import matplotlib.pyplot as plt

# Import data:
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int,
              'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': float, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

sales = pd.read_csv('kc_house_data_small.csv', dtype=dtype_dict)

train = pd.read_csv('kc_house_data_small_train.csv', dtype=dtype_dict)
valid = pd.read_csv('kc_house_data_validation.csv', dtype=dtype_dict)
test = pd.read_csv('kc_house_data_small_test.csv', dtype=dtype_dict)

# 5. Using get_numpy_data (or equivalent), extract numpy arrays of the training, test, and validation sets.
features_list_str = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
                     'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
                     'sqft_living15', 'sqft_lot15']

features, output = lr.get_numpy_data(train, features_list_str, 'price')
features_valid, output_valid = lr.get_numpy_data(valid, features_list_str, 'price')
features_test, output_test = lr.get_numpy_data(test, features_list_str, 'price')

# 6. In computing distances, it is crucial to normalize features.
# Otherwise, for example, the ‘sqft_living’ feature (typically on the order of thousands)
# would exert a much larger influence on distance than the ‘bedrooms’ feature (typically on the order of ones).
# We divide each column of the training feature matrix by its 2-norm, so that the transformed column has unit norm.
features, norms = lr.normalize_features(features)
features_valid = features_valid / norms
features_test = features_test / norms

# Compute a single distance
# 7. To start, let's just explore computing the “distance” between two given houses.
# We will take our query house to be the first house of the test set and look at the distance between
# this house and the 10th house of the training set.

# To see the features associated with the query house,
# print the first row (index 0) of the test feature matrix.
# You should get an 18-dimensional vector whose components are between 0 and 1.
# Similarly, print the 10th row (index 9) of the training feature matrix.

print(features_test[0])
print(features[9])

# 8. Quiz Question: What is the Euclidean distance between the query house and the 10th house of the training set?
#                                         _________________________________________
# Euclidean distance: distance(xi,xq) = √(xj[1] − xq[1])² + ... + (xj[d] − xq[d])²

euclidean_distance = np.sqrt(np.sum((features[9] - features_test[0]) ** 2))
print(euclidean_distance)

# Note: Do not use the ‘np.linalg.norm’ function; use ‘np.sqrt’, ‘np.sum’, and the power operator (**) instead.
# The latter approach is more easily adapted to computing multiple distances at once.


# 9. Of course, to do nearest neighbor regression,
# we need to compute the distance between our query house and all houses in the training set.

# To visualize this nearest-neighbor search, let's first compute the distance from our query house (features_test[0])
# to the first 10 houses of the training set (features_train[0:10]) and then search for the nearest neighbor
# within this small set of houses. Through restricting ourselves to a small set of houses to begin with,
# we can visually scan the list of 10 distances to verify that our code for finding the nearest neighbor is working.

# Write a loop to compute the Euclidean distance from the query house to each
# of the first 10 houses in the training set.

query_house = features_test[0]
distance_dict = {}
for k in range(0, 10):
    distance_dict[k] = np.sqrt(np.sum((features[k] - query_house ** 2)))
    print(k, ':', round(distance_dict[k], 3))

# 10. Quiz Question: Among the first 10 training houses, which house is the closest to the query house?
print(min(distance_dict.items(), key=lambda dis: dis[1]))
# 0, 0.32122952832339524 First House

# 11. It is computationally inefficient to loop over computing distances to all houses in our training dataset.
# Fortunately, many of the numpy functions can be vectorized, applying the same operation over
# multiple values or vectors. We now walk through this process. (The material up to #13 is specific to numpy;
# if you are using other languages such as R or Matlab, consult relevant manuals on vectorization.)

# Consider the following loop that computes the element-wise difference between the features of
# the query house (features_test[0]) and the first 3 training houses (features_train[0:3]):
# print features_train[0:3] - features_test[0]

# Perform 1-nearest neighbor regression
# 12. Now that we have the element-wise differences, it is not too hard to compute the Euclidean distances
# between our query house and all of the training houses. First, write a single-line expression to define a variable
# ‘diff’ such that ‘diff[i]’ gives the element-wise difference between the features of the query house and the i-th
# training house.

diff = features - features_test[0]
# To test the code above, run the following cell, which should output a value -0.0934339605842:
diff[-1].sum()

# 13. The next step in computing the Euclidean distances is to take these feature-by-feature
# differences in ‘diff’, square each, and take the sum over feature indices.
# That is, compute the sum of squared feature differences for each training house (row in ‘diff’).

# computes this sum of squared feature differences for all training houses. Verify that the two expressions
np.sum(diff ** 2, axis=1)[15]
# and
np.sum(diff[15] ** 2)
# yield the same results.
# That is, the output for the 16th house in the training set is equivalent to having examined only the 16th
# row of ‘diff’ and computing the sum of squares on that row alone.
# verified.

# 14. With this result in mind, write a single-line expression to compute the Euclidean
# distances from the query to all the instances. Assign the result to variable distances.

diff = features - features_test[0]
euclidean_distance = np.sqrt(np.sum(diff ** 2, axis=1))

# Hint: distances[100] should contain 0.0237082324496.
print(euclidean_distance[100])

# 15. Now you are ready to write a function that computes the distances from a query house to all training houses.
# The function should take two parameters: (i) the matrix of training features and (ii)
# the single feature vector associated with the query.
from learnt.regression.knn import compute_distances

euclidean_distance = compute_distances(features, features_test[0])
print(euclidean_distance[100])

# 16. Quiz Question: Take the query house to be third house of the test set (features_test[2]).
# What is the index of the house in the training set that is closest to this query house?

euclidean_distance = compute_distances(features, features_test[2])
print('index', np.argmin(euclidean_distance), ': ', min(euclidean_distance))

# 17. Quiz Question: What is the predicted value of the query house based on 1-nearest neighbor regression?

print(output[np.argmin(euclidean_distance)])

# Perform k-nearest neighbor regression
# 18. Using the functions above, implement a function that takes in
#
# the value of k;
# the feature matrix for the instances; and
# the feature of the query
# and returns the indices of the k closest training houses.
# For instance, with 2-nearest neighbor, a return value of [5, 10] would indicate that the 6th and 11th
#   training houses are closest to the query house.

from learnt.regression.knn import k_nearest_neighbors

four_neighbors = k_nearest_neighbors(4, features, features_test[2])

# 19. Quiz Question: Take the query house to be third house of the test set (features_test[2]).
#   What are the indices of the 4 training houses closest to the query house?

print(four_neighbors)

# 20. Now that we know how to find the k-nearest neighbors,
# write a function that predicts the value of a given query house.
# For simplicity, take the average of the prices of the k nearest neighbors in the training set.
# The function should have the following parameters:

# 21. Quiz Question: Again taking the query house to be third house of the test set (features_test[2]),
# predict the value of the query house using k-nearest neighbors with k=4 and the simple averaging method
# described and implemented above.

from learnt.regression.knn import predict_output_of_query

prediction_third_house = predict_output_of_query(4, features, output, features_test[2])
print('prediction for the 3rd house: ', prediction_third_house)

# 23. Quiz Question: Make predictions for the first 10 houses in the test set, using k=10.
#   What is the index of the house in this query set that has the lowest predicted value?
#   What is the predicted value of this house?

first10houses = features_test[:10]
first10houses_predict = lr.knn_predict_output(10, features, output, first10houses)

print(first10houses_predict)
print(np.argmin(first10houses_predict))

# Choosing the best value of k using a validation set
# 24. There remains a question of choosing the value of k to use in making predictions.

max_k = 16
rss = np.zeros(max_k - 1)
for k in range(1, max_k):
    valid_predict = lr.knn_predict_output(k, features, output, features_valid)
    rss[k - 1] = lr.get_residual_sum_squares(output_valid, valid_predict)  # k start at 1

# lets see how k is:
plt.plot(range(1, max_k), rss, '.-')
plt.xlabel('K')
plt.ylabel('RSS')
plt.show()

print('k equal to,', np.argmin(rss) + 1, 'with rss: ', np.min(rss), ', produced lowest rss')

# Quiz Question: What is the RSS on the TEST data using the value of k found above?
#   To be clear, sum over all houses in the TEST set.

predict = lr.knn_predict_output(8, features, output, features_test)
print('RSS on test with k=8 is :', lr.get_residual_sum_squares(output_test, predict))
