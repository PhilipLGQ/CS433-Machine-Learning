# costs.py

from helpers import *
import numpy as np


def compute_mse(e):
    return np.mean(e**2) / 2


def compute_loss_mse(y, tx, w):
    e = y - tx.dot(w)
    loss = compute_mse(e)
    return loss

def sigmoid(inx):
    #print(inx.shape,type(inx))
    output = []
    #print(inx.shape)
    for i in range (len(inx)):
        #print(inx[i].shape)
        if inx[i]>=0:
            output.append(1.0/(1+np.exp(-inx[i])))
        else:
            output.append(np.exp(inx[i])/(1+np.exp(inx[i])))
    output = np.asarray(output).reshape((-1,1))
    return output

def calculate_loss_logistic(y, tx, w):
    #print()
    sigmoid_pred = sigmoid(tx.dot(w))
    loss = -(y.T.dot(np.log(sigmoid_pred))
             + (1 - y).T.dot(np.log(1 - sigmoid_pred)))
    loss = np.squeeze(loss)

    return loss
