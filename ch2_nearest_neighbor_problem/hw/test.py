import numpy as np
import matplotlib.pyplot as plt
import bst as bst
    
def main():
    data_size = 10
    data = np.random.permutation(data_size).tolist()

    plt.figure(figsize = (15,15), dpi = 80)
    plt.xlim((-3,12))
    plt.title('visualization of data set')

    for i,d in enumerate(data):
        plt.scatter(i, d, c = 'red', s = 30, marker = 'x', alpha = 0.7)
    plt.show()

    root = bst.build_bst_recursively(data)
    '''root2 = bst.build_bst_iteratively(data)

    bst.inorder(root)
    bst.inorder(root2)'''

    near = bst.one_nn_search(root, 4.4)
    print(near)

if __name__ == '__main__':
    main()