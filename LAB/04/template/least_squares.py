# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    LHS = tx.T.dot(tx)
    RHS = tx.T.dot(y)
    w = np.linalg.solve(LHS,RHS)
    
    err = y - tx.dot(w)
    loss_mse = err.T.dot(err)/(2*len(y))

    return w, loss_mse
