import numpy as np
import kdtree 
import datetime

def main():
    # configuration
    db_size = 1000000
    dim = 3
    leaf_size = 1
    k = 8

    db_np = np.random.rand(db_size, dim)
    print("-------Start construct kd tree-------")
    start_time = datetime.datetime.now()
    root = kdtree.kdtree_construction(db_np, leaf_size=leaf_size)
    end_time = datetime.datetime.now()
    print("Construct a kd-tree costs: ", ((end_time - start_time).seconds * 1e6 + (end_time - start_time).microseconds) / 1e6, " seconds")
    
    depth = [0]
    max_depth = [0]
    kdtree.traverse_kdtree(root, depth, max_depth)
    print("tree max depth: %d" % max_depth[0])

    print("------KNN Search-------")
    query = np.asarray([0, 0, 0])
    start_time = datetime.datetime.now()
    result_set = kdtree.KNNResultSet(capacity=k)
    kdtree.kdtree_knn_search_lecture(root, db_np, result_set, query)
    end_time = datetime.datetime.now()
    print("KNN Search costs: ", ((end_time - start_time).seconds * 1e6 + (end_time - start_time).microseconds) / 1e6, " seconds")
    
    print(result_set)
    
    print("------Brute force search------")
    start_time = datetime.datetime.now()
    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    print(nn_idx[0:k])
    print(nn_dist[0:k])
    end_time = datetime.datetime.now()
    print("Brute Force KNN Search costs: ", ((end_time - start_time).seconds * 1e6 + (end_time - start_time).microseconds) / 1e6, " seconds")
    
    print("------Radius search-------")
    query = np.asarray([0, 0, 0])
    start_time = datetime.datetime.now()
    result_set = kdtree.RadiusNNResultSet(radius = 0.5)
    kdtree.kdtree_radius_search_lecture(root, db_np, result_set, query)
    end_time = datetime.datetime.now()
    print("RNN Search costs: ", ((end_time - start_time).seconds * 1e6 + (end_time - start_time).microseconds) / 1e6, " seconds")
    # print(result_set)
 

if __name__ == '__main__':
    main()