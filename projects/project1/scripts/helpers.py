# helpers.py

from costs import *
from preprocess import *
# from cross_validation import *
import numpy as np


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
        Batch iteration function used for Stochastic Gradient Descent.
        Arguments: y (labels)
                   tx (feature matrix)
                   batch_size (scale of a batch)
                   shuffle (Shuffle the feature matrix and labels if True)
    """
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
    """
        Polynomial expansion of preprocessed feature matrix, up to (degree - 1).
        Arguments: x (preprocessed feature matrix)
                   degree (the upper limit of polynomial expansion, upper limit not included)

    """
    poly = np.ones((len(x), 1))
    for degrees in range(1, degree):
        poly = np.c_[poly, np.power(x, degrees)]

    return poly


def compute_gradient(y, tx, w):
    """
        Compute the gradient of loss function.
        Arguments: y (labels)
                   tx (feature matrix)
                   w (weight)
    """
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(y)
    return grad, e


def sigmoid(t):
    """
        Sigmoid function used in logistic regression, np.exp overflow prevented, returns sigmoid output
        Arguments: t (list of variables in sigmoid function)
    """
    output = []
    for i in range(len(t)):
        if t[i] >= 0:
            output.append(1.0 / (1.0 + np.exp(-t)))
        else:
            output.append(np.exp(t[i]) / (1 + np.exp(t[i])))
    output = np.asarray(output).reshape((-1, 1))
    return output


def calculate_gradient_logistic(y, tx, w):
    """
        Calculate gradient of loss function in logistic regression
        Arguments: y (labels relabeled as 0 and 1)
                   tx (feature matrix)
                   w (weights)
    """
    sigmoid_pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(sigmoid_pred - y)
    return grad


def calculate_hessian(y, tx, w):
    """
        Calculate hessian in logistic regression
        Arguments: tx (feature matrix)
                   w (weights)
    """
    sigmoid_pred = sigmoid(tx.dot(w))

    # Generate a diagonal matrix for values of sigmoid_pred
    # Use its transpose form to generate a 1D array of sigmoid prediction values
    sigmoid_pred = np.diag(sigmoid_pred.T[0])

    # Use element-wise multiplication to get the diagonal matrix S
    S = np.multiply(sigmoid_pred, (1 - sigmoid_pred))

    # Calculate the hessian of L(w)
    hessian = tx.T.dot(S).dot(tx)

    return hessian


def learning_by_SGD_logistic(y, tx, w, gamma, batch_size=1):
    """
        Logistic regression by stochastic gradient descent, return weight and loss
        Arguments: y (labels relabeled as 0 and 1)
                   tx (feature matrix)
                   w (weights)
                   gamma (Gamma parameter)
                   batch_size (number of samples in batch)
    """
    for tx_batch, y_batch in batch_iter(y, tx, batch_size, num_batches=1):
        loss = calculate_loss_logistic(y_batch, tx_batch, w)
        grad = calculate_gradient_logistic(y_batch, tx_batch, w)
        w = w - gamma * grad

    return w, loss


def penalized_logistic_regression(y, tx, w, lambda_):
    """
        Calculate loss, gradient, hessian for regularized logistic regression
        Arguments: y (labels relabeled as 0 and 1)
                   tx (feature matrix)
                   w (weights)
                   lambda_ (Lambda parameter)
    """
    # return loss, gradient, and hessian
    loss = np.squeeze(calculate_loss_logistic(y, tx, w) + lambda_ * (w.T.dot(w)))
    grad = calculate_gradient_logistic(y, tx, w) + 2 * lambda_ * w
    hessian = calculate_hessian(y, tx, w) + 2 * lambda_

    return loss, grad, hessian


def learning_by_penalized_logistic(y, tx, w, gamma, lambda_, batch_size=1):
    """
        Regularized logistic regression by stochastic gradient descent, return weight and loss
        Arguments: y (labels relabeled as 0 and 1)
                   tx (feature matrix)
                   w (weights)
                   gamma (Gamma parameter)
                   lambda_ (Lambda parameter)
                   batch_size (number of samples in batch)
    """
    for tx_batch, y_batch in batch_iter(y, tx, batch_size, num_batches=1):
        loss, grad, _ = penalized_logistic_regression(y_batch, tx_batch, w, lambda_)
        w = w - gamma * grad

    return w, loss
