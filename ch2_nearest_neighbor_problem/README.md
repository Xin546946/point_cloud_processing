# ch2 Nearest neighbors

这次作业分为两部分，KDTree和Octree。
## 1. KDTree
* 先给出**scipy.spatial**库提供的对bin文件kdtree建树(leaf_size = 32)，knn(k = 8), rnn(r = 1)的结果，以及brute force搜索的结果(brute / knn = 17.5)
~~~
sklearn kdtree ----------
Kdtree: build 499.153ms, knn 1.025ms, radius 1.278ms, brute 17.959ms
~~~
* 再给出**open3d**库提供的相应函数的结果(注:open3d的KDTree没找到leafsize的接口,所以不太清楚这里默认的leaf_size默认是多少),所以无法比较，接下来就用scipy.spatial函数来做reference比较
~~~
open3d kdtree ----------
Kdtree: build 91.094ms, knn 1.024ms, radius 0.021ms
~~~
* 再给出用老师代码实现的kdtree的结果(brute / knn = 1.63)
~~~
My kdtree ------------
Kdtree: build 584.774ms, knn 20.512ms, radius 2.063ms, brute 33.524ms
~~~

* 下面开始研究加速<br>
***思路1***: 在建树的时候，对所有点排序，选取中值需要大量计算，然而，我们只关注中间的点到底是什么，至于中值前后是否排序与我无关。对于c++有nth_element可以调用，python貌似没找到这种api，所以想试试如果随机选取百分之10的点，用这些点来排序，那么建树的时间至少会大量降低。分析一下，这样可能稍微降低了kdtree的质量，但是建树应该会变快。
~~~
如果每次只选择百分之10的点进行排序，则结果跑如下(brute / knn = 1.79),可见，没有显著提升
My kdtree -------------
Kdtree: build 940.622ms, knn 8.816ms, radius 0.559ms, brute 15.883ms
~~~ 