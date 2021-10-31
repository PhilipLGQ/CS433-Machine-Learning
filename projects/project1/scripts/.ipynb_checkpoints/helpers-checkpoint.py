# helpers.py

from costs import *
from preprocess import *
from cross_validation import *
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


def find_optimal(y, tx, degrees, k_fold, lambdas, seed=1):
    # Split the data into k-fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # Set lists for collecting best lambda & rmse for each degree
    best_lambda = []
    best_rmse = []
    
    for degree in degrees:
        rmse_val = []
        
        for lambda_ in lambdas:
            rmse_val_lambda_ = []
            
            for k in range(k_fold):
                _, loss_val, w = cross_validation(y, tx, k_indices, k, lambda_, degree)
                rmse_val_lambda_.append(loss_val)

            rmse_val.append(np.mean(rmse_val_lambda_))
        
        index_opt_lambda = np.argmin(rmse_val)
        best_lambda.append(lambdas[index_opt_lambda])
        best_rmse.append(rmse_val[index_opt_lambda])
    
    opt_degree = degrees[np.argmin(best_rmse)]
    opt_lambda = best_lambda[np.argmin(best_rmse)]
    
    return opt_degree, opt_lambda

def find_optimal_KMC(y, tx, degrees, k_fold, lambdas, k_clusters, seed=1):
    # Split the data into k-fold
    k_indices = build_k_indices(y, k_fold, seed)
    x_k = tx.copy()

    # Set lists for collecting best lambda & rmse for each degree
    best_lambda = []
    best_rmse = []
    best_k = []
    best_degree = []

    for k_mean in k_clusters:
        
        filehandler = open(f'result_{k_mean}.txt', 'a')
        
        x_k = tx.copy()
        k_cluster = 5 * k_mean + 5

        k_means_replacing(x_k, k_cluster)
        x_k = standardize(x_k)

        for degree in degrees:
            rmse_val = []

            for _lambda in lambdas:
                rmse_val_lambda = []

                for k in range(k_fold):
                    _, loss_val, w = cross_validation(y, x_k, k_indices, k, _lambda, degree)
                    rmse_val_lambda.append(loss_val)

                filehandler.write("No. of clusters {}\n".format(k_cluster))
                filehandler.write("lambda {}\n".format(_lambda))

                filehandler.write("loss {}\n".format(np.mean(rmse_val_lambda)))
                filehandler.write("degree {}\n".format(degree))
                filehandler.write("\n\n")

                rmse_val.append(np.mean(rmse_val_lambda))

            index_opt_lambda = np.argmin(rmse_val)
            best_lambda.append(lambdas[index_opt_lambda])
            best_rmse.append(rmse_val[index_opt_lambda])
            best_k.append(k_mean)
            best_degree.append(degree)

    opt_degree = best_degree[np.argmin(best_rmse)]
    opt_lambda = best_lambda[np.argmin(best_rmse)]
    opt_k = best_k[np.argmin(best_rmse)]
    

    filehandler.close()

    return opt_degree, opt_lambda, opt_k



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


def learning_by_SGD_logistic(y, tx, w, gamma, batch_size=1):
    for tx_batch, y_batch in batch_iter(y, tx, batch_size, num_batches=1):
        loss = calculate_loss_logistic(y_batch, tx_batch, w)
        grad = calculate_gradient_logistic(y_batch, tx_batch, w)
        w = w - gamma * grad

    return w, loss


def penalized_logistic_regression(y, tx, w, lambda_):
    # return loss, gradient, and hessian
    loss = np.squeeze(calculate_loss_logistic(y, tx, w) + lambda_ * (w.T.dot(w)))
    grad = calculate_gradient_logistic(y, tx, w) + 2 * lambda_ * w
    hessian = calculate_hessian(y, tx, w) + 2 * lambda_

    return loss, grad, hessian


def learning_by_penalized_logistic(y, tx, w, gamma, lambda_, batch_size=1):
    for tx_batch, y_batch in batch_iter(y, tx, batch_size, num_batches=1):
        loss, grad, _ = penalized_logistic_regression(y_batch, tx_batch, w, lambda_)
        w = w - gamma * grad

    return w, loss
