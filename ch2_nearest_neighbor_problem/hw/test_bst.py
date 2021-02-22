import numpy as np
import matplotlib.pyplot as plt
from bst import construct_bst, insert
from bst import preorder, inorder, postorder

# generate test data 
data_size = 10
# randomly permute a sequence, or return a permuted range
data = np.random.permutation(data_size).tolist() 
data = np.array([1, 1, 1, 2, 3, 5, 3, 6, 4, 5, 6, 7, 8])
data = np.array([1, 4, 7, 6, 3, 13, 14, 10, 8])
print(data)

# visualize datas
plt.figure(figsize = (15,15))
plt.xlim((-3,12))
plt.title('visualization of data set')

for i,d in enumerate(data):
    plt.scatter(i, d, c = 'red', s = 30, marker = 'x', alpha = 0.7)

# plt.show()

# apply BST

bst_node = None
bst_root = construct_bst(data, 'iterative')
preorder(bst_root)
print("---------")
inorder(bst_node)
print("---------")
postorder(bst_node)
