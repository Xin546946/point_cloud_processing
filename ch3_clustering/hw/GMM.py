# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math
from KMeans_components import compute_min_dist_through_datas_and_centers

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

def vis_2d_gmm(samples, weights, means, covs, title):
    """Visualizes the model and the samples"""
    plt.figure(figsize=[7,7])
    plt.title(title)
    plt.scatter(samples[:, 0], samples[:, 1], label="Samples", c=next(plt.gca()._get_lines.prop_cycler)['color'])

    for i in range(means.shape[0]):
        c = next(plt.gca()._get_lines.prop_cycler)['color']

        (largest_eigval, smallest_eigval), eigvec = np.linalg.eig(covs[i])
        phi = -np.arctan2(eigvec[0, 1], eigvec[0, 0])

        plt.scatter(means[i, 0:1], means[i, 1:2], marker="x", c=c)

        a = 2.0 * np.sqrt(largest_eigval)
        b = 2.0 * np.sqrt(smallest_eigval)

        ellipse_x_r = a * np.cos(np.linspace(0, 2 * np.pi, num=200))
        ellipse_y_r = b * np.sin(np.linspace(0, 2 * np.pi, num=200))

        R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
        r_ellipse = np.array([ellipse_x_r, ellipse_y_r]).T @ R
        plt.plot(means[i, 0] + r_ellipse[:, 0], means[i, 1] + r_ellipse[:, 1], c=c,
                 label="Component {:02d}, Weight: {:0.4f}".format(i, weights[i]))
    plt.legend()
    plt.draw()


def gaussian_log_density(samples: np.ndarray, mean: np.ndarray, covariance: np.ndarray):
    dim = mean.shape[0]
    chol_covariance = np.linalg.cholesky(covariance)
    # Efficient and stable way to compute the log determinant and squared term efficiently using the cholesky
    logdet = 2 * np.sum(np.log(np.diagonal(chol_covariance) + 1e-25))
    # (Actually, you would use scipy.linalg.solve_triangular but I wanted to spare you the hustle of setting
    #  up scipy)
    chol_inv = np.linalg.inv(chol_covariance)
    exp_term = np.sum(np.square((samples - mean) @ chol_inv.T), axis=-1)
    return -0.5 * (dim * np.log(2 * np.pi) + logdet + exp_term)


def init_centers(datas, k):
    center = random.choice(datas)
    index = np.argwhere(datas == center)[0][0]
    init_centers_idx = []
    init_centers_idx.append(index)
    # np.random.seed(0)
    init_centers_datas = []
    init_centers_datas.append(center)
    
    flag = k-1 
    
    while(flag):
        
        distances = compute_min_dist_through_datas_and_centers(init_centers_datas, datas)
        # print(distances)
        center = np.random.choice(distances, p = distances.ravel())
        index = np.argwhere(distances == center)[0][0]
        init_centers_datas.append(datas[index])
        init_centers_idx.append(index)
        flag -= 1

    return init_centers_idx

class GMM(object):
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.means = None
        self.covs = None
        self.weights = None
        self.prev_means = None
        self.center_trajectory = []
    
    # 屏蔽开始
    # estep computes p(z|x) using old model from previoud iteration
    def e_step(self,samples: ndarray): # 
        # compute p(x|z)
        densities = []
        for i in range(len(self.weights)):
            densities.append(np.exp(gaussian_log_density(samples, self.means[i], self.covs[i])))
        densities = np.stack(densities, -1) # list to array (2000,3)

        # compute p(x,z) = p(x|z)p(z)
        joint_densities = densities * self.weights[None,...]

        # compute p(z|x) = p(x,z) / p(x) = p(x,z) / sum_z p(x,z)
        responsibilities = joint_densities / np.sum(joint_densities, -1, keepdims = True)

        return responsibilities


    def m_step(self, samples, responsibilities):

        self.prev_means = self.means

        # update weights p(z) = 1/ N sum_x p(z|x)
        unnormalized_weights =np.sum(responsibilities, axis = 0)
        self.weights = unnormalized_weights / len(samples)

        # Compute update using weighted maximum likelihood
        sample_weights = responsibilities / unnormalized_weights[None, :]  # = np.sum(responsibilities, axis=0)

        weighted_samples = sample_weights[...,None] * samples[:,None,:] # shape: [N, num_components, dim]
        self.means = np.sum(weighted_samples, axis = 0) # shape: [n_clustes, dim]
        
        # Compute update using weighted maximum likelihood.
        diffferences = samples[:, None] - self.means[None] # shape [N, n_clusters]
        outer_products = diffferences[:,:,None] * diffferences[...,None]
        weighted_outer_products = sample_weights[..., None, None] * outer_products   # shape: [N, num_components, dim, dim]
        self.covs = np.sum(weighted_outer_products, axis=0)   # shape: [num_components, dim, dim]
    # 屏蔽结束
    

    def init_gmm(self, samples):
        # init_idx = np.random.choice(len(samples), self.n_clusters, replace=False)
        init_idx = init_centers(samples, self.n_clusters)
        self.means = samples[init_idx]
        self.covs = np.tile(np.eye(samples.shape[-1])[None, ...], [self.n_clusters, 1, 1])
        self.weights = np.ones(self.n_clusters) / self.n_clusters
        

    def fit(self, samples: ndarray, vis = None):
        # 作业3
        # 屏蔽开始
        self.init_gmm(samples)

        for i in range(self.max_iter):
            responsibilities = self.e_step(samples) # given: p(x|z), solve: p(z|x) = p(x|z) * p(z) / sum_z (p(x,z))
            self.m_step(samples, responsibilities) # given p(z)
            if i % 5 == 0 and vis != None:
                vis_2d_gmm(samples, self.weights, self.means, self.covs, title ="After Itaration {:02d}".format(i))
            if i != 0 and np.linalg.norm(np.asarray(self.prev_means) - np.asarray(self.means)) < 0.01:
                break
        # 屏蔽结束
    
    def predict(self, samples):
        # 屏蔽开始
        responsibilities = self.e_step(samples)
        return np.argmax(responsibilities,axis=1)
        # 屏蔽结束

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    # plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    samples = generate_X(true_Mu, true_Var)
    gmm = GMM(n_clusters=3)
    gmm.fit(samples, vis = True)
    # plt.show()
    
    cat = gmm.predict(samples)
    print(cat)
    i = 0
    plt.scatter(samples[:,0], samples[:,1], cat)
    plt.show()
    

