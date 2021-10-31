import numpy as np
from numpy.linalg import norm


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


def std_norm_preprocess(tx, func='std'):
    if func == 'std':
        tx_std = standardize(tx)
        return tx_std

    elif func == 'norm':
        tx_norm = normalize(tx)
        return tx_norm


def split_reformat_data(feature, label, id, k_arr=[]):

    f_arr = []
    l_arr = []

    f0 = np.delete(feature[feature[:, 22] == 0], np.r_[4:7, 12, 22:30], axis=1)
    f1 = np.delete(feature[feature[:, 22] == 1], np.r_[4:7, 12, 22, 26:29], axis=1)
    f2 = np.delete(feature[feature[:, 22] == 2], np.r_[22], axis=1)
    f3 = np.delete(feature[feature[:, 22] == 3], np.r_[22], axis=1)
    f_arr.extend([f0, f1, f2, f3])

    l0, l1 = label[feature[:, 22] == 0], label[feature[:, 22] == 1]
    l2, l3 = label[feature[:, 22] == 2], label[feature[:, 22] == 3]
    l_arr.extend([l0, l1, l2, l3])

    _ids = np.concatenate((id[feature[:, 22] == 0], id[feature[:, 22] == 1],
                               id[feature[:, 22] == 2], id[feature[:, 22] == 3]))
    if len(k_arr) == 0:
        return f_arr, l_arr, _ids

    else:
        for idx, (fea, n_clu) in enumerate(zip(f_arr, k_arr)):
            k_means_replacing(fea, n_clu)
            f_arr[idx] = standardize(fea)
        return f_arr, l_arr, _ids


def data_preprocess(tx, y, id, k_list=[], replacing='lr', mode='std'):
    tx_split, y_split, id_split = split_reformat_data(tx, y, id)

    if replacing in ('mean', 'median', 'lr', 'zero'):
        for i in range(len(tx_split)):
            replace_missing_value(tx_split[i], 0, func=replacing)

            # Standardization or normalization
            tx_split[i] = std_norm_preprocess(tx_split[i], mode)

        return tx_split, y_split, id_split

    if replacing == 'k_means':
        if len(k_list) == 0:
            return tx_split, y_split, id_split

        else:
            for idx, (fea, n_clu) in enumerate(zip(tx, k_list)):
                k_means_replacing(fea, n_clu)

                # Standardization or normalization
                tx_split[idx] = std_norm_preprocess(tx_split[idx], mode)

            return tx_split, y_split, id_split


def replace_missing_value(data, col=0, func='lr'):
    if func == 'median':
        rep = np.median(data[data[:, col] != -999][:, col])
        data[:, col] = np.where(data[:, col] == -999, rep, data[:, col])

    elif func == 'zero':
        rep = 0
        data[:, col] = np.where(data[:, col] == -999, rep, data[:, col])

    elif func == 'mean':
        rep = np.mean(data[data[:, col] != -999][:, col])
        data[:, col] = np.where(data[:, col] == -999, rep, data[:, col])

    elif func == 'lr':
        normal_x = np.mat(np.delete(data[data[:, 0] != -999], np.r_[0], axis=1))
        normal_y = np.mat(data[data[:, 0] != -999][:, 0]).T
        abnormal_x =  np.mat(np.delete(data[data[:, 0] == -999], np.r_[0], axis=1))

        if np.linalg.det(normal_x.T * normal_x) != 0:
            weight = (normal_x.T * normal_x).I * (normal_x.T * normal_y)
            predict_y = abnormal_x * weight
            data[np.where(data[:, 0] == -999)[0]] = np.c_[predict_y, abnormal_x]

        else:
            print("Singular Matrix!")


def initialise_centroids(data, n_clusters, seed=5):
    
    # As we meant to have reproduce identical result
    # We specify a fixed seed for random
    np.random.seed(seed)
    indices = np.random.choice(data.shape[0], size=n_clusters)
    centroids = np.copy(data[indices])
    
    return centroids[:, 1:]


def compute_distance(data, n_clusters, centroids):
        distance = np.zeros((data.shape[0], n_clusters))
        for k in range(n_clusters):
            
            # Here, we select columns inside the function
            # Thus not necessary to adjust the pass-in data
            row_norm = norm(data[:, 1:] - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance
    
    
def find_closest_cluster(distance):
        return np.argmin(distance, axis=1)
    
    
def compute_centroids(data, old_closest_clusters, n_clusters):
    
    # Create an empty numpy array to store computed centroids
    centroids = np.zeros((n_clusters, data.shape[1] - 1))
    
    for k in range(n_clusters):
        centroids[k, :] = np.mean(data[old_closest_clusters == k, 1:], axis=0)
    return centroids
    
    
def k_means_clustering(data, col, n_clusters, max_iter):
    
    centroids = initialise_centroids(data, n_clusters)
    
    for i in range(max_iter):
        distance = compute_distance(data, n_clusters, centroids)
        clusters = find_closest_cluster(distance)

        # The concat of first column value and its corresponding cluster
        stats = np.c_[data[:,col], clusters]

        # Store centriods for reference
        old_centroids = centroids
        
        # Update centroids
        centroids = compute_centroids(data, stats[:,1], n_clusters)

        # Break if no further updates
        if np.all(old_centroids == centroids):
            break
            
    return stats


def k_means_replacing(data, n_clusters, max_iter=200, mode='mean', col=0, seed=5):
    
    # Make a copy of data to perform substitution
    data_copy = np.copy(data)
    
    stats = k_means_clustering(data, col, n_clusters, max_iter)
    
    # Subset data with not nan value
    # As we already know the shape of stats, we will use hard-coded index
    not_nan = stats[stats[:,0] != -999]
    
    # Calculate means over each cluster with valid data
    if mode=="mean":
    
        for clu in np.unique(stats[:,1]):
            rep_cluster = np.round(not_nan[:,0].mean(where=not_nan[:,1]==clu), 3)
            stats[ np.where((stats[:,0] == -999) & (stats[:,1] == clu)), 0] = rep_cluster
        
    # Replace value accordingly in the original data
    data_copy[:,col] = stats[:,0]
    
    return data_copy