import numpy as np
import matplotlib.pyplot as plt
import bst as bst
import copy
import datetime
    
def main():
    data_size = 1000000
    data = np.random.permutation(data_size).tolist()
    #data = [1,4,7,6,3,13,14,10,8]

    '''plt.figure(figsize = (15,15), dpi = 80)
    plt.xlim((-3,12))
    plt.title('visualization of data set')

    for i,d in enumerate(data):
        plt.scatter(i, d, c = 'red', s = 30, marker = 'x', alpha = 0.7)
    plt.show()'''
    start_time = datetime.datetime.now()
    root = bst.build_bst_recursively(data)
    end_time = datetime.datetime.now()
    print("building tree costs ",((end_time - start_time).seconds * 1e6 + (end_time - start_time).microseconds) / 1e3, " ms")
    #root2 = bst.build_bst_iteratively(data)

    #bst.inorder(root)
    #bst.inorder(root2)
    start_time = datetime.datetime.now()
    near = bst.KNNsearch(root, 500, 5)
    end_time = datetime.datetime.now()
    
    print("search costs ",((end_time - start_time).seconds * 1e6 + (end_time - start_time).microseconds) / 1e3, " ms")
    print(near)

    '''class ob:
        def __init__(self,fi,se):
            self.fi = fi
            self.se = se
    
    a = ob(1,2)
    b = copy.deepcopy(a)
    b.fi = 4
    print(a.fi)'''

if __name__ == '__main__':
    main()