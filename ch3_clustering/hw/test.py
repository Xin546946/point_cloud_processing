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



p = np.array([1,5,4,3,8])
s = np.argsort(p)[:2]
what = p[s]

x1 = p[:,:,None]
x2 = p[...,None]

outer_products = p[:,:,None] * p[...,None]

a = np.array([1,2])
ax = np.expand_dims(a,axis = 0)
b = np.array([[1,2],[3,4]])
c = a * b
c2 = ax * b
print(c)