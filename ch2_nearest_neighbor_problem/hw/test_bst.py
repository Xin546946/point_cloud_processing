import numpy as np
import matplotlib.pyplot as plt
import datetime

from bst import construct_bst
from bst import preorder, inorder, postorder
from bst import onenn_search

from brut_force import brute_force_onenn_search
# generate test data 
data_size = 1000000
# randomly permute a sequence, or return a permuted range
print("-------Generate 1000000 datas--------")
data = np.random.permutation(data_size).tolist()
print("-------Generate 1000000 datas finished--------") 
# data = np.array([1, 1, 1, 2, 3, 5, 3, 6, 4, 5, 6, 7, 8])
# data = np.array([1, 4, 7, 6, 3, 13, 14, 10, 8])
# print(data)

# visualize datas
# plt.figure(figsize = (15,15))
# plt.xlim((-3,12))
# plt.title('visualization of data set')

# for i,d in enumerate(data):
#     plt.scatter(i, d, c = 'red', s = 30, marker = 'x', alpha = 0.7)

# plt.show()

# apply BST

bst_node = None
print("------ Start to build a tree ------")
start_time = datetime.datetime.now()
bst_root = construct_bst(data, 'recursive')
end_time = datetime.datetime.now()
print("Construct a tree costs: ", ((end_time - start_time).seconds * 1e6 + (end_time - start_time).microseconds) / 1e6, " seconds")
# preorder(bst_root)
# print("---------")
# inorder(bst_root)
# print("---------")
# postorder(bst_root)

query_data = np.random.choice(1000000)

print("------Search via one nn search----------")
start_time = datetime.datetime.now()
onenn_node = onenn_search(bst_root, query_data)
end_time = datetime.datetime.now()
print("Search {} from 1000000 datas costs: {} milliseconds".format(query_data, (end_time - start_time).microseconds / 1e3 ))

print("------Search via brute force search--------")
start_time = datetime.datetime.now()
min_dist_data = brute_force_onenn_search(data, query_data)
end_time = datetime.datetime.now()
print("Search {} from 1000000 datas costs: {} milliseconds".format(query_data, (end_time - start_time).microseconds / 1e3))

