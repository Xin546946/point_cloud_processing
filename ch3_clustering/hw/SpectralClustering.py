#build graph to get adjacency matrix
#compute unnormalized laplacian L
#compute the first smallest k eigenvectors of v1,v2,v3 of L
#let V n*k be the matrix containing the vectors v1,v2,v3 of as columns
#for i = 1,...n, let yi be the vector corresponding to the i-th row of V
#Cluster the points{yi} with k-means algorithm into clusterC1..Ck
#the final output clusters A1,...Ak
import numpy as np
from sklearn import cluster, datasets
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

#step1
def buildGraph(data):
    n = len(data)
    adjacencyMatrix = np.zeros((n,n))
    tree = KDTree(data, leafsize=8)
    for i in range(n):
        neighbors = tree.query_ball_point(data[i], r = 0.2)##matters a lot
        diff = np.linalg.norm(data[i] - data[neighbors], axis = 1)
        adjacencyMatrix[i,neighbors] = diff

    return adjacencyMatrix
#step2
def computeFeatures(adjacencyMatrix, k):
    Dvec = np.sum(adjacencyMatrix, axis = 1)
    D = np.diag(Dvec)
    L = D - adjacencyMatrix

    eigenvalues, eigenvectors = np.linalg.eig(L)
 
    eigenvectors = eigenvectors[:,np.argsort(eigenvalues)[:2]]
    return eigenvectors


if __name__ == '__main__':
    #x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples = n_samples, factor = 0.5, noise = 0.05)
    X,label = noisy_circles

    adj = buildGraph(X)
    ys = computeFeatures(adj,2)
    #print(ys)

    kmeans = cluster.KMeans(n_clusters=2)
    kmeans.fit(ys)
    color =  ['b','r']
    for i in range(X.shape[0]):
        plt.scatter(X[i][0], X[i][1], c = color[kmeans.labels_[i]], s=2)

    plt.show()
    print("done")



            
