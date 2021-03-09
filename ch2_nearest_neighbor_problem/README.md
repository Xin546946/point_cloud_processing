# ch2 Nearest neighbors

这次作业分为两部分，KDTree和Octree。
## 1. KDTree
* 先给出**scipy.spatial**库提供的对bin文件kdtree建树(leaf_size = 32)，knn(k = 8), rnn(r = 1)的结果，以及brute force搜索的结果(brute / knn = 17.5)
~~~
scipy kdtree ----------
Kdtree: build 499.153ms, knn 1.025ms, radius 1.278ms, brute 17.959ms
~~~
* 再给出**open3d**库提供的相应函数的结果(注:open3d的KDTree没找到leafsize的接口,所以不太清楚这里默认的leaf_size默认是多少),所以无法比较，接下来就用scipy.spatial函数来做reference比较
~~~
open3d kdtree ----------
Kdtree: build 91.094ms, knn 1.024ms, radius 0.021ms
~~~
* 再给出用老师代码实现的kdtree的结果(brute / knn = 1.11)
~~~
My kdtree ------------
Kdtree: build 283.742ms, knn 11.855ms, radius 0.505ms, brute 13.218ms
~~~

