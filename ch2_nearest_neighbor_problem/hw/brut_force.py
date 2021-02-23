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

def main():
    data = np.random.permutation(1000)
    min_dist_data = brute_force_onenn_search(data, 100)
    print(min_dist_data)

if __name__ == '__main__':
    main()