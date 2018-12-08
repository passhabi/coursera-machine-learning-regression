from learnt.regression import get_residual_sum_squares
import pandas as pd


def k_fold_cross_validation(k: int, l2_penalty: float, data: pd.DataFrame, output: pd.DataFrame):
    from sklearn.linear_model import Ridge

    n = len(output)  # number of data points (observations)
    valid_error = 0
    for i in range(k):
        start = int((n * i) / k)  # start index
        end = int((n * (i + 1)) / k)  # end index

        # split data to valid set and train set:
        x_valid_set = data[start:end + 1]  # omitted segment for testing training set
        #   same for the output array:
        y_valid_set = output[start:end + 1]

        #   assume valid_set with - and train segments with + and [] for selecting area.
        x_train_set = data[0:start]  # [+++]---++++++++++
        x_train_set = x_train_set.append(data[end + 1:])  # [+++]---[++++++++++]
        #   same again for output:
        y_train_set = output[0:start]
        y_train_set = y_train_set.append(output[end + 1:])

        # convert to numpy array:
        features_train = x_train_set.iloc[:, :].values
        output_train = y_train_set.iloc[:].values
        #   do the same for validation set
        features_valid = x_valid_set.iloc[:, :].values
        output_valid = y_valid_set.iloc[:].values

        # TODO: use my future ridge regression method instead of skiti-learn
        model = Ridge(l2_penalty, normalize=True)
        model.fit(features_train, output_train)

        # compute RSS for validation set:
        prediction = model.predict(features_valid)
        # store currant RSS:
        valid_error += get_residual_sum_squares(output_valid, prediction)

    return valid_error / k