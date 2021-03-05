# 第三章作业 聚类算法

1. **K Means**
    <br>
* K Means 的基本思想如下，假设需要聚K类：
~~~ pseudocode
  1. 在所给定的数据中随机选取K个点作为中心点
  2. while(不满足终止条件):
     1. 固定中心点，更新数据集每个点的label
     2. 固定每个点的label，更新聚类中心
~~~

* 根据上述逻辑，可以发现每个数据具有两个attributes, feature & label，所以我写了一个小class，Sample用于存储每个点以及它对应的label
~~~ python
class Sample(object):
    def __init__(self, data, label = -1):
        self.data = data
        self.label = label
~~~

* 根据伪代码逻辑，代码框架如下：
  ~~~ python
    #  
    self.samples = [(Sample(data)) for data in datas]
    self.centers = init_center(datas, self.k_)
    tolerance = 1e10
    iteration = 0
    while(tolerance > self.tolerance_ and iteration < self.max_iter_):
        
        iteration += 1
        self.samples = update_label(self.samples, self.centers)
        
        last_centers,centers = update_center(self.centers,self.samples,self.k_)

        tolerance = compute_distance(last_centers, centers)
        
        print("Iteration : {}, Tolerance : {}".format(iteration, tolerance))
  ~~~

* 每一个函数的实现细节可以参考KMeans_components.py文件。

* 跑benchmark之前，先用自己的测试函数可视化一下，一些bug也是在这个小的测试数据上跑的时候找到的

<center class="half">
    <img src="./figure/fig1.png" width="280"/><img src="./figure/fig2.png" width="280"/>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.3 K Means的结果</div>
</center>
* 由于初始化完全随机，所以无法保证聚类质量。
