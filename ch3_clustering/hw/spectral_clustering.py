import numpy as np
from sklearn import cluster, datasets
import matplotlib.pyplot as plt
from scipy.spatial import KDTree 
from KMeans import K_Means

def compute_weighted_matrix(datas, leaf_size = 8, radius = 1.0):
    num_points, dim = datas.shape
    weighted_matix = np.zeros((num_points, num_points))
    kdtree = KDTree(datas, leafsize=leaf_size)
    
    for i in range(num_points):
        rnn_data_idx = kdtree.query_ball_point(datas[i], r=radius)
        rnn_datas = datas[rnn_data_idx]
        diff = np.linalg.norm(datas[i] - rnn_datas, axis=1)
        weighted_matix[i , rnn_data_idx] = diff
    
    return weighted_matix

def compute_degree_matrix(weighted_matix):
    degree_vector = np.sum(weighted_matix, axis=1)
    degree_matrix = np.diag(degree_vector)
    return degree_matrix


def spectral_clustering(datas, n_clusters = 2):
    # import pdb; pdb.set_trace()
    weighted_matrix = compute_weighted_matrix(X, radius=0.2)
    degree_matrix = compute_degree_matrix(weighted_matrix)
    laplacian_matrix = degree_matrix - weighted_matrix
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
    kth_eigenvector = eigenvectors[:,np.argsort(eigenvalues)[:n_clusters]]
    plt.scatter(kth_eigenvector[:,0],kth_eigenvector[:, 1],s=2)
    plt.show()
    k_means = cluster.KMeans(n_clusters=n_clusters)
    k_means.fit(kth_eigenvector)
    return k_means.labels_

if __name__ == '__main__':
    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    X, label = noisy_circles
    
    color =  ['b','r']
    
    plt.figure(figsize=(7,7))
    labels = spectral_clustering(X)
    for i in range(X.shape[0]):
        plt.scatter(X[i][0], X[i][1], c = color[labels[i]], s=2)
    
    # plt.show()
