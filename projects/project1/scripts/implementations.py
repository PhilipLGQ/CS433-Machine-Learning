# implementation.py

# This file applies all required ML methods in Project1

# Included methods:
# least_squares_GD, least_squares_SGD, least_squares, ridge_regression,
# logistic_regression, and reg_logistic_regression

# Return type: (w, loss) of the last iteration step


from costs import *
import numpy as np


def least_squares_GD(y, tx, initial_w, max_iters, gamma):

    losses = []
    tx = np.c_[np.ones((tx.shape[0], 1)), tx]

    w = initial_w
    ws = [initial_w]

    for n_iter in range(max_iters):
        grad, _ = compute_gradient(y, tx, w)
        loss = compute_loss_mse(y, tx, w)
        w = w - grad * gamma

        ws.append(w)
        losses.append(loss)

    return ws[-1], losses[-1]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1):
    losses = []
    tx = np.c_[np.ones((tx.shape[0], 1)), tx]

    w = initial_w
    ws = [initial_w]

    for n_iter in range(max_iters):
        for tx_batch, y_batch in batch_iter(y, tx, batch_size, num_batches=1):
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            loss = compute_loss_mse(y_batch, tx_batch, w)
            w = w - grad * gamma
            
            ws.append(w)
            losses.append(loss)

        print('loss={a}, iteration={b}'.format(a=losses[-1], b=n_iter))
                                                
    return ws[-1], losses[-1]


def least_squares(y, tx):
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    LHS = tx.T.dot(tx) + 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    RHS = tx.T.dot(y)

    # Solve the linear system for ridge w*
    w = np.linalg.solve(LHS, RHS)

    # Here we calculate root mse as loss
    loss = np.sqrt(2 * compute_loss_mse(y, tx, w))

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, threshold, ):
    losses = []
    tx = np.c_[np.ones((tx.shape[0], 1)), tx]

    w = initial_w
    ws = [initial_w]

    for i in range(max_iters):
        w, loss = learning_by_SGD_logistic(y, tx, w, gamma)
        losses.append(loss)
        ws.append(w)
        
        # log info
        if i % 100 == 0:
            print("Current iteration={a}, loss={b}".format(a=i, b=losses[-1]))
            
        # Converge Criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return ws[-1], losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, threshold):
    losses = []
    w = initial_w
    ws = [initial_w]

    for i in range(max_iters):
        w, loss = learning_by_penalized_logistic(y, tx, w, gamma, lambda_)
        losses.append(loss)
        ws.append(w)

        if i % 100 == 0:
            print("Current iteration={}".format(i))

        # Converge Criterion
        if len(losses) > 1:
            if np.abs(losses[-1] - losses[-2]) < threshold:
                break

    return ws[-1], losses[-1]
