# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
from helpers import *
import csv
import numpy as np
from implementations import *
from cross_validation import *



def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def generate_weights(tx, y, best_d, best_l, best_k, ids):
    """
        Generate weights for ridge regression, return weights
        Arguments: tx (preprocessed feature matrix)
                   y (labels)
                   best_d, best_l, best_k (optimal degree, lambda, and k-value learned)
                   ids (original id list)
    """
    weights = []
    feature_arr, label_arr, = tx, y
    for idx, (f, l) in enumerate(zip(feature_arr, label_arr)):
        # Polynomial Feature Transform
        poly_feature = build_poly(f, best_d[idx])

        # Training ridge regression on the entire training
        w, _= ridge_regression(l, poly_feature, best_l[idx])
        weights.append(w)

    return weights


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def reformat_result(pred, re_ids):
    """
        Reorder the prediction result according to the original id order.
        Arguments: pred (prediction list for
                   re_ids (original id list)
    """
    pred_pair = [(i, j) for i, j in zip(re_ids, pred)]
    result = [j for _, j in sorted(pred_pair)]
    
    return result


def data_pred(tx, w, re_ids, poly=False, best_d=[]):
    """
        Generate the list of predicted labels, return to test_data_reordered function for output.
        Arguments: tx (preprocessed feature matrix)
                   w (trained weights)
                   re_ids (separated id list from preprocessing function)
                   poly (Polynomial expansion to best degree if True)
                   best_d = (default empty, should be provided when poly = True)
    """
    pred = []
    for idx, (f, weight) in enumerate(zip(tx, w)):
        if poly:
            tX = build_poly(f, best_d[idx])
        else:
            tX = np.c_[np.ones((len(f), 1)), f]
        pred.extend(predict_labels(weight, tX))

    return reformat_result(pred, re_ids)


def metric_pred(pred, y):
    """
        Returns an accuracy used for evaluating the performance on training set, round to 4 decimal places
        Arguments: pred (list with predicted labels)
                   y (list with original labels)
    """
    count = 0
    for i in range(pred):
        if count[i] == y[i]:
            count += 1

    accuracy = round((count / len(y)), 4)

    return accuracy


def find_optimal(y, tx, degrees, k_fold, lambdas, seed=1):
    """
        Calculate to find the optimal degree and optimal lambda, when method is not k-means.
        Arguments: y (labels)
                   tx (preprocessed feature matrix)
                   degrees (pre-set degree range to find the best)
                   lambdas (pre-set lambda range to find the best)
                   seed (default random seed)
    """
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
    """
        Select the best number of clusters, best degrees and best lambdas for k-means imputed logistic regression.
        Arguments: y (labels)
                   tx (splitted feature matrix)
                   degrees (pre-set degree range to select the best)
                   k_fold (number of folds)
                   lambdas (pre-set degree range to select the best)
                   k_clusters (pre-set cluster index to select the best, k = [10 * k_clusters + 5])
                   seed (default random seed)
    """
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
        k_cluster = 10 * k_mean + 5

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


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
