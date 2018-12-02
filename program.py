from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from learnt import regression as lr

# import data:
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int,
              'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': float, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

data = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
train = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
test = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)



