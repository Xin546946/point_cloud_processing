import numpy as np
import random
from scipy.stats import multivariate_normal
'''mean = np.array([1,1])
cov = np.array([[10,0],[0,10]])
print(mean)
print(cov)
rv = multivariate_normal(mean,cov)
pos = np.array([1,1])
p = rv.pdf(pos)
print(p)'''
x = np.array([1,2,3])
ex = np.exp(x)

p = np.array([[1,2,3],[2,7,4],[3,10,5],[5,9,7]])
sumq = np.sum(p, axis = 0)
what = sumq[None,:]

x1 = p[:,:,None]
x2 = p[...,None]

outer_products = p[:,:,None] * p[...,None]

a = np.array([1,2])
ax = np.expand_dims(a,axis = 0)
b = np.array([[1,2],[3,4]])
c = a * b
c2 = ax * b
print(c)