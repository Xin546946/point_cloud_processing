##Nearest Neighbor Problem

---------------

1. **kdtree**
    ~~~python
    # 作业1
    # 屏蔽开始
    middle_left_index = math.ceil(point_indices_sorted.shape[0] / 2) - 1
    middle_left_point_index = point_indices_sorted[middle_left_index]
    middle_left_point_value = db[middle_left_point_index]

    middle_right_index = middle_left_index + 1
    middle_right_point_index = point_indices_sorted[middle_right_index]
    middle_right_point_value = db[middle_right_point_index]

    root.value = (middle_right_point_value + middle_left_point_value) / 2
    root.left = kdtree_recursive_build(root.left,
                                        db,
                                        point_indices_sorted[0:middle_right_index],
                                        axis_round_robin(axis, dim = db.shape[1]),
                                        leaf_size)
    root.right = kdtree_recursive_build(root.right,
                                        db,
                                        point_indices_sorted[middle_right_index:],
                                        axis_round_robin(axis, dim = db.shape[1]),
                                        leaf_size)
    # 屏蔽结束

    # 作业2
    # 提示：仍通过递归的方式实现搜索
    # 屏蔽开始
    if query[root.axis] >= root.value[root.axis]:
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value[root.axis]) < result_set.worstDist():
            kdtree_knn_search(root.left, db, result_set, query)
    else:    
        kdtree_knn_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value[root.axis]) < result_set.worstDist():
            kdtree_knn_search(root.right, db, result_set, query)
    # 屏蔽结束

    # 作业3
    # 提示：通过递归的方式实现搜索
    # 屏蔽开始
    if query[root.axis] >= root.value[root.axis]:
        kdtree_radius_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value[root.axis]) < result_set.worstDist():
            kdtree_radius_search(root.left, db, result_set, query)
    else:    
        kdtree_radius_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value[root.axis]) < result_set.worstDist():
            kdtree_radius_search(root.right, db, result_set, query)
    # 屏蔽结束

* with leaf_size = 32 , k = 8, builing tree costs 111.762ms, knn search 