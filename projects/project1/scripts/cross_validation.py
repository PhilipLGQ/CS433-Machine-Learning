# cross-validation.py


from helpers import *
from implementations import *
import numpy as np



def build_k_indices(y, k_fold, seed):
    """
        Split range of indices for k-fold cross validation.
        Arguments: y (labels)
                   k_fold (number of folds)
                   seed (random seed)
    """
    num_rows = len(y)
    interval = int(num_rows / k_fold)

    random_indices = np.random.permutation(num_rows)
    k_indices = [random_indices[k * interval: (k + 1) * interval] for k in range(k_fold)]

    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """
        K-fold cross validation, return the training loss, validation loss (loss in RMSE),
        and weight of ridge regression.
        Arguments: y (labels)
                   x (preprocessed feature matrix)
                   k_indices (splitted k-fold indices)
                   k (current k-fold iteration index)
                   lambda_ (current lambda value used for ridge regression)
                   degree (current degree used for polynomial expansion)
    """

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
    w, _ = ridge_regression(y_tr, poly_tx_tr, lambda_)

    # Calculate loss of cost function root mse (RMSE)
    loss_tr = np.sqrt(2 * compute_loss_mse(y_tr, poly_tx_tr, w))
    loss_te = np.sqrt(2 * compute_loss_mse(y_te, poly_tx_te, w))

    return loss_tr, loss_te, w


def build_poly(x, degree):
    poly = np.ones((len(x), 1))
    for degrees in range(1, degree):
        poly = np.c_[poly, np.power(x, degrees)]

    return poly