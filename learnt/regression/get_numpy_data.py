import numpy as np


def get_numpy_data(data_frame, features_str, output_str):
    """
    data_frame preparation
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
