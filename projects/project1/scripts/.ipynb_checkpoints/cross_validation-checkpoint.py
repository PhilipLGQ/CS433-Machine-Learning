# cross-validation.py


from helpers import *
from implementations import *
import numpy as np


def build_k_indices(y, k_fold, seed):
    num_rows = len(y)
    interval = int(num_rows / k_fold)

    random_indices = np.random.permutation(num_rows)
    k_indices = [random_indices[k * interval: (k + 1) * interval] for k in range(k_fold)]

    return k_indices


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""

    # get k'th subgroup in test, others in train
    indice_te = k_indices[k]
    indice_tr = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    indice_tr = indice_tr.reshape(-1)
    x_tr = x[indice_tr]
    x_te = x[indice_te]
    y_tr = y[indice_tr]
    y_te = y[indice_te]

    # Form data with polynomial degree
    poly_tx_tr = build_poly(x_tr, degree)
    poly_tx_te = build_poly(x_te, degree)

    # Do ridge regression
    w = ridge_regression(y_tr, poly_tx_tr, lambda_)

    # Calculate loss of cost function root mse (RMSE)
    loss_tr = np.sqrt(2 * compute_mse(y_tr, poly_tx_tr, w))
    loss_te = np.sqrt(2 * compute_mse(y_te, poly_tx_te, w))

    return loss_tr, loss_te, w