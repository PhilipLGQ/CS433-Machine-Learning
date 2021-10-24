# helpers.py

from costs import *
import numpy as np


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)

    if shuffle:
        indices = np.random.permutation(np.arange(data_size))
        tx_shuffled = tx[indices]
        y_shuffled = y[indices]

    else:
        tx_shuffled = tx
        y_shuffled = y

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min(data_size, (batch_num + 1) * batch_size)
        if start_index != end_index:
            yield tx_shuffled[start_index:end_index], y_shuffled[start_index:end_index]


def build_poly(x, degree):
    poly = np.ones((len(x), 1))
    for degrees in range(1, degree):
        poly = np.c_[poly, np.power(x, degrees)]

    return poly


def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(y)
    return grad, e


def sigmoid(t):
    sigmoid = 1.0 / (1.0 + np.exp(-t))
    return sigmoid


def calculate_gradient_logistic(y, tx, w):
    sigmoid_pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(sigmoid_pred - y)
    return grad


def calculate_hessian(y, tx, w):
    sigmoid_pred = sigmoid(tx.dot(w))

    # Generate a diagonal matrix for values of sigmoid_pred
    # Use its transpose form to generate a 1D array of sigmoid prediction values
    sigmoid_pred = np.diag(sigmoid_pred.T[0])

    # Use element-wise multiplication to get the diagonal matrix S
    S = np.multiply(sigmoid_pred, (1 - sigmoid_pred))

    # Calculate the hessian of L(w)
    hessian = tx.T.dot(S).dot(tx)

    return hessian


def learning_by_GD_logistic(y, tx, w, gamma):
    loss = calculate_loss_logistic(y, tx, w)
    grad = calculate_gradient_logistic(y, tx, w)

    w = w - gamma * grad

    return w, loss


def penalized_logistic_regression(y, tx, w, lambda_):
    # return loss, gradient, and hessian
    loss = np.squeeze(calculate_loss_logistic(y, tx, w) + lambda_ * (w.T.dot(w)))
    grad = calculate_gradient_logistic(y, tx, w) + 2 * lambda_ * w
    hessian = calculate_hessian(y, tx, w) + 2 * lambda_

    return loss, grad, hessian


def learning_by_penalized_logistic(y, tx, w, gamma, lambda_):
    loss, grad, _ = penalized_logistic_regression(y, tx, w, lambda_)

    w = w - gamma * grad

    return w, loss