# Generate isotropic Gaussian blobs for clustering.
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from KMeans import K_Means

n_samples = 100
n_features = 2 # dim of the datas
n_clusters = 4 
random_state = 10 # 

X, y = make_blobs(n_samples=n_samples, n_features = n_features, centers = n_clusters, random_state = random_state)

fig = plt.figure(figsize = (8,8), dpi = 80, facecolor = 'w', edgecolor = 'k')


dictionary = {}

color_map = ['r', 'g', 'b', 'k']
for i in range(4):
    dictionary[i] = color_map[i]


k_means = K_Means(n_clusters = 4)
samples, centers = k_means.fit(X)

for sample in samples:
    plt.scatter(sample.data[0], sample.data[1], marker = 'x', c = dictionary[sample.label])

for center in centers:
    plt.scatter(center.data[0], center.data[1], marker = 'D', c = dictionary[center.label])

plt.title("Kmeans result visualization")
plt.xlabel("x")
plt.ylabel("y")
plt.show()



# for sample in samples:


plt.show()
