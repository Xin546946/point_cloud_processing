# NN-Trees
Python implementation of Binary Search Tree, kd-tree for tree building, kNN search, fixed-radius search.

~~~ python
   * given binary search tree
   * knn search
compare current node with query data
if query_data > current_node.value:
    
    knn_search in right direction
    if distance(query_data, current_node.value) < worst_dist 
    # which means left side has the chance
        knn_search in left direction
        add current_node
elif query_data < current_node.value:
    knn_search in left direction
    if distance(query_data, current_node.value) < worst_dist
        knn_search in right direction
else: # query_data == current_node.value
     knn search in left_direction
     knn search in right_direction
    
  
**very useful kdtree c++ code**
https://github.com/jlblancoc/nanoflann