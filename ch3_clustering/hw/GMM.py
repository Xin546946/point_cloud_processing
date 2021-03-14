# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

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



# def gaussian_pdf(data : ndarray, mean : ndarray, var: ndarray):
#     import pdb; pdb.set_trace()
#     data = np.expand_dims(data,0)
#     assert data.shape[0] == mean.shape[0] and data.shape[0] == var.shape[0] and data.shape[1] == 1
#     coeff_inv = np.power(2 * (math.pi), dim / 2.0) * np.power(np.linalg.det(var) + 1e-8, 0.5)
#     result = (1 / coeff_inv) * np.exp(-0.5 * (data - mean).T * np.linalg.inv(var + 1e-8) * (data - mean))
#     assert result.shape[0] == 1 and result.shape[1] == 1
#     return result
class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.means = None
        self.convs = None
        self.weights = None
    
    # 屏蔽开始
    # estep computes p(z|x) using old model from previoud iteration
    def e_step(self,samples: ndarray): # 
        # compute p(x|z)
        densities = []
        for i in range(len(self.weights)):
            densities.append(np.exp(gaussian_log_density(samples, self.mean[i], self.convs[i])))
        densities = np.stack(densities, -1)

        # compute p(x,z) = p(x|z)p(z)
        joint_densities = densities * self.weights[None,...]

        # compute p(z|x) = p(x,z) / p(x) = p(x,z) / sum_z p(x,z)
        responsibilities = joint_densities / np.sum(joint_densities, -1, keepdims = True)

        return responsibilities

    def m_step(samples, responsibilities):



    # 屏蔽结束
    
    def init_gmm(self, samples):
        init_idx = np.random.choice(len(samples), self.n_clusters, replace=False)
        self.means = samples[init_idx]
        self.convs = np.tile(np.eye(samples.shape[-1])[None, ...], [self.n_clusters, 1, 1])
        self.weights = np.ones(self.n_clusters) / self.n_clusters
        

    def fit(self, samples):
        # 作业3
        # 屏蔽开始
        init_gmm(samples)

        for i in range(self.max_iter):
            responsibilities = e_step(samples) # given: p(x|z), solve: p(z|x) = p(x|z) * p(z) / sum_z (p(x,z))
            m_step(samples, responsibilities) # given p(z)
        
        # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        pass
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
    plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    samples = generate_X(true_Mu, true_Var)
    import pdb; pdb.set_trace()
    gmm = GMM(n_clusters=3)
    gmm.fit(samples)
    cat = gmm.predict(samples)
    print(cat)
    # 初始化

    

