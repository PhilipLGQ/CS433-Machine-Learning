# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

# Calculate the Min Square Error in terms of e
def mse_calculation(e_term):
    return 1/2*np.mean(e_term**2)

# Calculate the Min Absolute Error in terms of e
def mae_calculation(e_term):
    return np.mean(np.abs(e_term))

# Compute the loss of given w on x & y
def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE / MAE
    e_term = y - tx.dot(w)
    loss = mse_calculation(e_term)
    return loss
    # ***************************************************
    # raise NotImplementedError