import pandas as pd


def polynomial_dataframe(feature: pd.DataFrame.axes, degree: int):
    """
    Generate polynomial features up to degree 'degree'.


    :param feature: Is pandas.Series, float, double type
    :param degree: to power
    :return: data frame to the degree power
    """
    # assume that degree >= 1
    # initialize the data frame:

    poly_df = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_df['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree + 1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_df[name] = feature ** power

    return poly_df
