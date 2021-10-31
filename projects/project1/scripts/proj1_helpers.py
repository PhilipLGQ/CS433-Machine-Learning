# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
from helpers import *
import csv
import numpy as np



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
    pred_pair = [(i, j) for i, j in zip(re_ids, pred)]
    result = [j for _, j in sorted(pred_pair)]
    
    return result


def data_pred(tx, w, ids, poly=False, best_d=[]):
    pred = []
    for idx, (f, weight) in enumerate(zip(tx, w)):
        if poly:
            tX = build_poly(f, best_d[idx])
        else:
            tX = np.c_[np.ones((len(f), 1)), f]
        pred.extend(predict_labels(weight, tX))

    return reformat_result(pred, ids)


def metric_pred(pred, y):
    count = 0
    for i in range(len(pred)):
        if pred[i] == y[i]:
            #print('!')
            count = count + 1
    accuracy = count / len(y)
    return accuracy


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
