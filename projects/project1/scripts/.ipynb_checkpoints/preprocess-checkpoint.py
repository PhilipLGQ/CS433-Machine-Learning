import numpy as np


def standardize(tx):
    tx_T = np.transpose(tx)
    tx_T_std = np.zeros((tx.shape[1], tx.shape[0]))

    for i in range(tx.shape[1]):
        tx_T_std[i] = (tx_T[i] - np.mean(tx_T[i])) / np.std(tx_T[i])

    return np.transpose(tx_T_std)


def normalize(tx):
    tx_T = np.transpose(tx)
    tx_T_norm = []

    for i in range(tx.shape[1]):
        tx_T_norm[i] = (np.max(tx_T[i], axis=0) - tx_T[i]) / \
                       (np.max(tx_T[i], axis=0) - np.min(tx_T[i], axis=0))

    return np.transpose(tx_T_norm)


def split_reformat_feature(feature):
    
    f_arr = []
    
    f0 = np.delete(feature[feature[:,22] == 0], np.r_[4:7, 12, 22:30], axis=1)
    f1 = np.delete(feature[feature[:,22] == 1], np.r_[4:7, 12, 22, 26:29], axis=1)
    f2 = np.delete(feature[feature[:,22] == 2], np.r_[22], axis=1)
    f3 = np.delete(feature[feature[:,22] == 3], np.r_[22], axis=1)
    f23 = np.concatenate((f2, f3))
    
    f_arr.append(f0)    
    f_arr.append(f1)
    f_arr.append(f23)
    
    # Replacing the missing values of first feature column with median
    # Standardize data
    for idx, elem in enumerate(f_arr):
        replace_missing_value(elem, 0, 'median')
        f_arr[idx] = standardize(elem)
    
    return f_arr


def split_label(feature, label):
    l_arr = []
    
    l0, l1 = label[feature[:,22] == 0], label[feature[:,22] == 1]
    l23 = np.concatenate((label[feature[:,22] == 2], label[feature[:,22] == 3]))
    
    l_arr.append(l0)    
    l_arr.append(l1)
    l_arr.append(l23)
    
    return l_arr


def reorder_id(feature, ID):
    _ids = np.concatenate((ID[feature[:,22] == 0], ID[feature[:,22] == 1],
                           ID[feature[:,22] == 2], ID[feature[:,22] == 3]))
    
    return _ids


def split_reformat_train(feature, label):
    
    f_arr = split_reformat_feature(feature)
    l_arr = split_label(feature, label)

    return f_arr, l_arr


def split_reformat_test(feature):
    
    f_arr = split_reformat_feature(feature)

    return f_arr


def replace_missing_value(data, col, func='median'):
    if func == 'median':
        rep = np.median(data[data[:, col] != -999][:, col])

    elif func == 'mean':
        rep = np.mean(data[data[:, col] != -999][:, col])

    data[:, col] = np.where(data[:, col] == -999, rep, data[:, col])
