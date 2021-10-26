# costs.py

from helpers import *
import numpy as np


def compute_mse(e):
    return np.mean(e**2) / 2


def compute_loss_mse(y, tx, w):
    e = y - tx.dot(w)
    loss = compute_mse(e)
    return loss


def calculate_loss_logistic(y, tx, w):
    sigmoid_pred = sigmoid(tx.dot(w))
    loss = -(y.T.dot(np.log(sigmoid_pred))
             + (1 - y).T.dot(np.log(1 - sigmoid_pred)))
    loss = np.squeeze(loss)

    return loss
