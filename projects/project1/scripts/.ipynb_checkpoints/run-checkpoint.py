# Useful starting lines
%matplotlib inline
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2

# Import files to use in preprocessing and machine learning
from implementations import *
from proj1_helpers import *
from preprocess import *
from cross_validation import *
from helpers import *
from costs import *


# Download train data and supply path here 
DATA_TRAIN_PATH = '../data/train.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


'''
*** Set hyper-parameters
'''

K_FOLD = 10
DEGREE = np.arange(3, 7)
DEGREE_jet_0 = np.arange(2, 4)
K_CLUSTERS = np.arange(1, 10)
SEED = 5
LAMBDA = np.logspace(-6, -2, 40)
METHOD = 'k_means'
MODE = 'std'


'''
*** Data preprocessing

TX: List with index 0, 1 & 2, each contains independent feature matrix for 
    events with 'PRI_jet_num' = 0, 1, 2&3 correspondingly;

Y: Array with index 0, 1 & 2, corresponding label vector (1 and -1) of TX;

r_ids: List with ids of each split's events in ascending order (for each split),
       later used in reordering;

replacing: ways to impute the missing value (-999), options: 'median', 'mean',
           'lr'(linear regression), 'k-means'(using ridge regression);
           
mode: ways to normalize the data, options: 'std' (standardization by mean and sd),
      'norm' (normalize data to 0 ~ 1), 'std_norm' (standardization followed by norm);         
'''

features, labels, reordered_ids = split_reformat_data(tX, y, ids)

ls_opt_degree = []
ls_opt_lambda = []
ls_opt_k = []


for idx in range (len(features)):
    
    # To save time, we are using a different range of degree for data with jet = 0
    if idx == 0:
        opt_d, opt_l, opt_k = find_optimal_KMC(labels[idx], features[idx], DEGREE_jet_0, K_FOLD, LAMBDA, K_CLUSTERS)
    else:   
        opt_d, opt_l, opt_k = find_optimal_KMC(labels[idx], features[idx], DEGREE, K_FOLD, LAMBDA, K_CLUSTERS)
    
    ls_opt_degree.append(opt_d)
    ls_opt_lambda.append(opt_l)    
    ls_opt_k.append(opt_k)
    
    

TX, Y, r_ids = data_preprocess(tX, y, ids, k_list = ls_opt_k, replacing=METHOD, mode=MODE)


y_pred = data_pred(TX, weights, r_ids, True, ls_opt_degree)


metric_pred(y_pred, y)


DATA_TRAIN_PATH = '../data/test.csv' 
test_y, test_tX, test_ids = load_csv_data(DATA_TRAIN_PATH)
test_tX, _, test_ids_pre = data_preprocess(test_tX, test_y, test_ids, k_list = ls_opt_k, replacing=METHOD, mode='std')
test_pred = data_pred(test_tX, weights, test_ids_pre, True, ls_opt_degree)
OUTPUT_PATH = 'data/pred.csv'
create_csv_submission(test_ids, test_pred, OUTPUT_PATH)




