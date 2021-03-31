import numpy as np

a = np.array([[1,2,3],[2,4,6]])
a = np.expand_dims(a,axis = 0)
print(a.shape)