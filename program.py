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
train = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
test = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)

# 9. Consider a simple model with 2 features: ‘sqft_living’ and ‘bedrooms’. The output is ‘price’.
# First, run get_numpy_data() (or equivalent) to obtain a feature matrix with 3 columns (constant column added).
# Use the entire ‘sales’ dataset for now.
# Normalize columns of the feature matrix. Save the norms of original features as ‘norms’.
# Set initial weights to [1,4,1].
# Make predictions with feature matrix and initial weights.
# Compute values of ro[i]
feature_matrix, output = lr.get_numpy_data(df, ['sqft_living', 'bedrooms'], 'price')
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
print(round(2 * ro[2]), ' up to ', round(2 * ro[1]))
# 161933396 up to 175878940

# 11. Quiz Question: What range of values of l1_penalty would set both w[1] and w[2] to zero,
# if we were to take a step in that coordinate?
print('grater than', round(2 * ro[1]))

