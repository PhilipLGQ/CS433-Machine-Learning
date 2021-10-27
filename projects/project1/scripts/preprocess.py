import numpy as np


def standardize(tx):
    tx_T = np.transpose(tx)
    tx_T_std = np.zeros((tx.shape[1], tx.shape[0]))

    for i in range(tx.shape[1]):
        tx_T_std[i] = (tx_T[i] - np.mean(tx_T[i])) / np.std(tx_T[i])

    return np.transpose(tx_T_std)


def normalize(tx):
    tx_T = np.transpose(tx)
    tx_T_norm = np.zeros((tx.shape[1], tx.shape[0]))

    for i in range(tx.shape[1]):
        tx_T_norm[i] = (np.max(tx_T[i]) - tx_T[i]) / \
                       (np.max(tx_T[i]) - np.min(tx_T[i]))

    return np.transpose(tx_T_norm)


def split_reformat_data(feature, label, id):
    f0 = np.delete(feature[feature[:,22] == 0], np.r_[4:7, 12, 22:30], axis=1)
    f1 = np.delete(feature[feature[:,22] == 1], np.r_[4:7, 12, 22, 26:29], axis=1)
    f2 = np.delete(feature[feature[:,22] == 2], np.r_[22], axis=1)
    f3 = np.delete(feature[feature[:,22] == 3], np.r_[22], axis=1)
    f23 = np.concatenate((f2, f3))

    l0, l1 = label[feature[:,22] == 0], label[feature[:,22] == 1]
    l23 = np.concatenate((label[feature[:,22] == 2], label[feature[:,22] == 3]))

    _ids = np.concatenate((id[feature[:,22] == 0], id[feature[:,22] == 1],
                           id[feature[:,22] == 2], id[feature[:,22] == 3]))

    return f0, f1, f23, l0, l1, l23, _ids


def split_reformat_test(feature, id):
    f0 = np.delete(feature[feature[:,22] == 0], np.r_[4:7, 12, 22:30], axis=1)
    f1 = np.delete(feature[feature[:,22] == 1], np.r_[4:7, 12, 22, 26:29], axis=1)
    f2 = np.delete(feature[feature[:,22] == 2], np.r_[22], axis=1)
    f3 = np.delete(feature[feature[:,22] == 3], np.r_[22], axis=1)
    f23 = np.concatenate((f2, f3))

    _ids = np.concatenate((id[feature[:,22] == 0], id[feature[:,22] == 1],
                           id[feature[:,22] == 2], id[feature[:,22] == 3]))

    return f0, f1, f23, _ids


def replace_missing_value(xx, col=0, algorithm = 'LinearRegression'):
    if algorithm == 'LinearRegression':
        normal_x = np.mat(np.delete(xx[xx[:, 0] != -999], np.r_[0], axis=1))
        normal_y = np.mat(xx[xx[:, 0] != -999][:, 0]).T
        abnormal_x =  np.mat(np.delete(xx[xx[:, 0] == -999], np.r_[0], axis=1))
        if (np.linalg.det(normal_x.T*normal_x) != 0):
            weight = (normal_x.T * normal_x).I * (normal_x.T * normal_y)
            predict_y = abnormal_x*weight
            xx[np.where(xx[:, 0] == -999)[0]] = np.c_[predict_y, abnormal_x]
        else:
            print("Singular Matrix!")

    if algorithm == 'Median':
        median = np.median(xx[xx[:, col] != -999][:, col])
        xx[:, col] = np.where(xx[:, col] == -999, median, xx[:, col])

    if algorithm == 'Mean':
        mean = np.mean(xx[xx[:, col] != -999][:, col])
        xx[:, col] = np.where(xx[:, col] == -999, mean, xx[:, col])

def k_means_cluster(data, y, k, max_iter=20):
    data = np.asarray(data, np.float32)
    indices = np.random.randint(0,data.shape[0],(1,k)).tolist()
    #print(indices)
    center = np.copy(data[indices])
    cluster = np.zeros(data.shape[0])
    for i in range(0,max_iter):
        one_hot1 = np.zeros(k*data.shape[0], np.float32)
        distance = np.sqrt(np.sum(np.square(np.expand_dims(data, axis=1) - center), axis=2))
        cluster = np.argmin(distance, axis=1)
        one_hot1[np.argmin(distance, axis=1) + np.arange(data.shape[0]) * k] = 1.
        one_hot2 = np.reshape(one_hot1, (data.shape[0], k))
        center = np.matmul(np.transpose(one_hot2, (1, 0)), data) / np.expand_dims(np.sum(one_hot2, axis=0), axis=1)     
    class_y = np.zeros((2, np.asarray(np.argmin(distance, axis=1)).shape[0]))
    class_y[0] = np.asarray(cluster)
    class_y[1] = np.asarray(y[0].T)[:,0]
    y_center = []
    for i in range(k):
        y_center.append(np.mean(class_y[1][class_y[0,:]==i]))
    return cluster, center, np.array(y_center)

def k_means_replacing(xx, k=11):
    normal_x = np.mat(np.delete(xx[xx[:, 0] != -999], np.r_[0], axis=1)) 
    abnormal_x = np.mat(np.delete(xx[xx[:, 0] == -999], np.r_[0], axis=1))
    normal_y = np.mat(xx[xx[:, 0]!= -999][:,0])
    cluster, center, y_center = k_means_cluster(normal_x,normal_y, k)
    replace_list = np.zeros((abnormal_x.shape[0],1))
    for j in range(abnormal_x.shape[0]):
        #if j%100 == 0:
            #print('Replacing '+str(j)+' out of'+str(abnormal_x.shape[0]))       
        tt = np.zeros((center.shape[0],center.shape[1]))
        for i in range(center.shape[0]):
            tt[i] = abnormal_x[j]
        ff = np.mat(tt-center)
        distance_matrix = np.array((ff*ff.T).diagonal()).T
        class_num = np.argmin(distance_matrix)
        replace_list[j] = y_center[class_num]
    replace_list = np.c_[replace_list,abnormal_x]
    xx[np.where(xx[:, 0]== -999)[0]] = replace_list
