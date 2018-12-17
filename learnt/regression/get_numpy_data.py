import numpy as np
from typing import List


def get_numpy_data(data_frame, features: List[str], output: str):
    """ Convert data frame to separate matrices.

    Function takes a data frame, a list of features (e.g. [‘sqft_living’, ‘bedrooms’]), to be
    used as inputs, and a name of the output (e.g. ‘price’).
    This function returns a features_matrix (2D array)
    consisting of first a column of ones followed by columns
    containing the values of the input features in the data set
    in the same order as the input list. It also return an output_array
    which is an array of the values of the output in the data set (e.g. ‘price’).
    data frame preparation

    :param data_frame:
        pandas.dataframe
    :param features:
        List[str] name of the features
    :param output:
        str name of the Y or output column in data frame
    :return: feature_matrix, output

    """
    if output not in data_frame:    # check if output column exists in data frame
        raise ImportError(output, 'is not exist in the data frame')

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
