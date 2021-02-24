import numpy as np


def brute_force_onenn_search(data, query_data):
    min_dist = float('inf')
    for d in data:
        if abs(d - query_data) < min_dist:
            min_dist = abs(d - query_data)
            min_dist_data = d
        elif d == query_data:
            min_dist_data = d
            return min_dist_data
        else:
            pass
    return min_dist_data
# todo brute force vs knn
def brute_force_knn_search(data, query_data, k):
    dist = query_data - data
    sorted_dist_index = np.argsort(dist)
    print(sorted_dist_index)
    print("---")
    print(dist)

def main():
    # test brute_force_onenn_search
    data = np.random.permutation(10)
    # min_dist_data = brute_force_onenn_search(data, 100)
    # print(min_dist_data)
    brute_force_knn_search(data, 10, 3)

if __name__ == '__main__':
    main()