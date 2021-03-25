import heapq as hq
import numpy as np

a = np.arange(3)
c = np.expand_dims(a,0)
b = np.array([[1,2,3],[2,3,4]])
po = a - b
ppp = c - b
d = np.linalg.norm(a-b, axis = 1)
e = np.linalg.norm(c-b, axis = 1)

print(a)