import numpy as np
import matplotlib.pyplot as plt
import datetime

from bst import construct_bst
from bst import preorder, inorder, postorder
from bst import onenn_search

from brut_force import brute_force_onenn_search

from bst import knn_search, radiusnn_search, knn_search_lecture, radius_search_lecture
from result_set import KNNResultSet, RadiusNNResultSet

def test_onenn_vs_brute_force(data_size, test = None):
    # generate test data 
    
    # randomly permute a sequence, or return a permuted range
    print("-------Generate {} datas--------".format(data_size))
    data = np.random.permutation(data_size).tolist()
    print("-------Generate {} datas finished--------\n".format(data_size)) 
    
    if test == 'preorder':
        preorder(bst_root)
    elif test == 'inorder':
        inorder(bst_root)
    elif test == 'postorder':
        postorder(bst_root)
    else:
        pass

    bst_node = None
    print("------ Start to build a tree ------")
    start_time = datetime.datetime.now()
    bst_root = construct_bst(data, 'recursive')
    end_time = datetime.datetime.now()
    print("Construct a tree costs: ", ((end_time - start_time).seconds * 1e6 + (end_time - start_time).microseconds) / 1e6, " seconds")
    query_data = np.random.choice(data_size)

    print("\n------Search via one nn search----------")
    start_time = datetime.datetime.now()
    onenn_node = onenn_search(bst_root, query_data)
    end_time = datetime.datetime.now()
    print("Search {} from {} datas costs: {} milliseconds".format(query_data, data_size, (end_time - start_time).microseconds / 1e3 ))

    print("\n------Search via brute force search--------")
    start_time = datetime.datetime.now()
    min_dist_data = brute_force_onenn_search(data, query_data)
    end_time = datetime.datetime.now()
    print("Search {} from {} datas costs: {} milliseconds".format(query_data, data_size, (end_time - start_time).microseconds / 1e3))


# test_onenn_vs_brute_force(100000, None)
def simple_test_data():
    return np.array([1, 4, 7, 6, 3, 13, 14, 10, 8])

print("------ Start to build a tree ------")
start_time = datetime.datetime.now()
test_data = np.random.permutation(10000000)
bst_root = construct_bst(test_data, 'recursive')
end_time = datetime.datetime.now()
print("Construct a tree costs: ", ((end_time - start_time).seconds * 1e6 + (end_time - start_time).microseconds) / 1e6, " seconds")



start_time = datetime.datetime.now()
result_set = radiusnn_search(bst_root, 5, 500)
end_time = datetime.datetime.now()
print("KNN SEARCH: ", ((end_time - start_time).seconds * 1e6 + (end_time - start_time).microseconds) / 1e6, " seconds")

print(result_set)
print("-----------------")
start_time = datetime.datetime.now()
result_set = RadiusNNResultSet(5)
radius_search_lecture(bst_root, 5, result_set)
end_time = datetime.datetime.now()
print("KNN SEARCH FROM LECTURE: ", ((end_time - start_time).seconds * 1e6 + (end_time - start_time).microseconds) / 1e6, " seconds")
print(result_set)

for i in result_set.dist_index_list:
    print(test_data[i.index])

