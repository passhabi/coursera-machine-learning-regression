import numpy as np
from typing import List


def get_numpy_data(data_frame, features: List[str], output: str):
    """
    data_frame preparation
    :param data_frame:
    :param features:
    :param output:
    :return: feature_matrix, output
    """
    data_frame = data_frame.copy()
    data_frame['constant'] = 1  # add a constant column to an SFrame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    # select the columns of data_frame given by the ‘features’ list into the Frame ‘features_frame’
    features_matrix = data_frame[features]
    # this will convert the features_frame into a numpy matrix
    features_matrix = np.array(features_matrix)
    # assign the column of data_frame associated with the target to the variable ‘output_array’
    output_array = data_frame[output]
    # this will convert the Array into a numpy array:
    output_array = np.array(output_array)
    return features_matrix, output_array
