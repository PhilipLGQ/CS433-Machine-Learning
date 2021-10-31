from cross_validation import *
from preprocess import *
from helpers import *
from implementations import *
from proj1_helpers import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load training dataset, divide into feature matrix, class labels, and event ids
DATA_TRAIN_PATH = '.data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


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

TX, Y, r_ids = data_preprocess(tX, y, ids, replacing='k_means ', mode='std_norm')


'''
*** Set optimal hyper-parameters derived from cross-validation
'''

