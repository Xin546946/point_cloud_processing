import numpy as np

alist = [21,23,35,17,31,57,31,15]
n = len(alist)
for i in range(n-1, 0, -1):
    count = 0
    for j in range(i):
        print(j)