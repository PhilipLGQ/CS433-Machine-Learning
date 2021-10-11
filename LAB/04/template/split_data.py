# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)

    # split the data based on the given ratio
    rows = len(y)
    
    # use random indices for data splitting
    r_index = np.random.permutation(rows)
    
    # calculate number of rows for training and testing
    r_train = round(ratio * len(y))
    r_test = round((1 - ratio) * len(y))
    
    # split the original dataset
    i_train = r_index[:r_train]
    i_test = r_index[r_train:]
    
    x_train = x[i_train]
    x_test = x[i_test]
    
    y_train = y[i_train]
    y_test = y[i_test]
    
    return x_train, x_test, y_train, y_test
