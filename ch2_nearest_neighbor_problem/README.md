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

## 2. Octree

* Octree 的结果

~~~
My octree --------------
Octree: build 11727.653ms, knn 0.981ms, radius 0.732ms, brute 17.305ms
~~~

## 3. 自己实现的部分

* 本想写一个template的kdtree，但是因为太不熟悉pcl库，中间出了点bug还没de出来，所以就调用了pcl的API看看效果先

~~~ C++

#include "tictoc.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

void kdtree_knn_search(pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud, int K,
                       std::vector<int> &neighbor_idx,
                       std::vector<float> &neighbor_distances) {

  pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
  tictoc::tic();
  kdtree.setInputCloud(point_cloud);
  std::cout << "Build KDtree costs: " << tictoc::toc() / 1e3 << "milliseconds"
            << '\n';
  pcl::PointXYZI searchPoint = point_cloud->points[0];
  // std::cout << "searching " << K << " nearest neighbors of point "
  //           << searchPoint << '\n';
  tictoc::tic();
  kdtree.nearestKSearch(searchPoint, K, neighbor_idx, neighbor_distances);
  std::cout << "8NN Search using KDtree costs: " << tictoc::toc() / 1e3
            << "milliseconds" << '\n';
  return;
}

int main(int argc, char **argv) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr(
      new pcl::PointCloud<pcl::PointXYZI>);

  std::string pcd_path = argv[1];
  if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_path, *cloud_ptr) ==
      -1) { // load the point cloud
    PCL_ERROR("Could not read file test_pcd.pcd. \n");
    return -1;
  } else {
    std::cout << "Loaded " << cloud_ptr->width * cloud_ptr->height
              << " data points from test_pcd.pcd. " << std::endl;
  }
  // for (size_t i = 0; i < cloud->points.size(); ++i)
  //   std::cout << "    " << cloud->points[i].x << " " << cloud->points[i].y
  //             << " " << cloud->points[i].z << std::endl;
  int k = 8;
  std::vector<int> neighbor_idx(k);
  std::vector<float> neighbor_distances(k);

  kdtree_knn_search(cloud_ptr, k, neighbor_idx, neighbor_distances);
}
~~~

~~~
Loaded 124669 data points from test_pcd.pcd. 
Build KDtree costs: 33.668milliseconds
8NN Search using KDtree costs: 0.031milliseconds
~~~

* 不过之前用opencv的图片测试了一些，还是可以看出效果的，早知道不应该作死写template，真的烧头发。。。

~~~C++
#pragma once
#include <algorithm>
#include <vector>

template <typename T>
struct GKDTreeNode {
    typedef GKDTreeNode* PtrNode;

    GKDTreeNode(T* data, int axis) : data_(data), axis_(axis) {
    }

    bool in_smaller_side(T* sample) {
        return (*sample)[axis_] < (*data_)[axis_];
    }

    bool in_larger_side(T* sample) {
        return (*sample)[axis_] > (*data_)[axis_];
    }

    T* data_;
    int axis_;

    PtrNode smaller_ = nullptr;
    PtrNode larger_ = nullptr;
    std::vector<T*> leaves_;
    bool has_leaves() {
        return (!this->leaves_.empty());
    }
};

template <typename T>
class GKdTree {
   public:
    typedef typename GKDTreeNode<T>::PtrNode PtrNode;
    GKdTree(T* head, int size, int leaf_size = 1);
    GKdTree() = default;
    ~GKdTree();

    std::vector<T*> rnn_search(T* data, typename T::DistType radius) const;

   private:
    void build_tree(PtrNode& curr, T* begin, T* end);
    void next_axis();

    PtrNode root_ = nullptr;

    int axis_ = 0;
    int leaf_size_;
    int size_;
};

/*--------------------------------------------------------
#####################implementation: GKdTree #####################
---------------------------------------------------------*/
template <typename T>
GKdTree<T>::GKdTree(T* head, int size, int leaf_size) : leaf_size_(leaf_size), size_(size) {
    this->build_tree(root_, head, head + size_);
}

template <typename T>
inline void GKdTree<T>::next_axis() {
    axis_ = (axis_ == T::dim_ - 1) ? 0 : axis_ + 1;
}

template <typename T>
void GKdTree<T>::build_tree(PtrNode& curr, T* begin, T* end) {
    int dist = std::distance(begin, end);
    T* mid = begin + dist / 2;

    std::nth_element(begin, mid, end, [=](const T& lhs, const T& rhs) { return lhs[axis_] < rhs[axis_]; });
    curr = new GKDTreeNode<T>(mid, axis_);

    next_axis();

    if (dist <= leaf_size_) {
        curr->leaves_.reserve(leaf_size_);
        std::for_each(begin, end, [&](T& data) { curr->leaves_.push_back(&data); });
    } else {
        build_tree(curr->smaller_, begin, mid);
        build_tree(curr->larger_, mid, end);
    }
}

template <typename T>
void inorder(GKDTreeNode<T>* curr, std::vector<GKDTreeNode<T>*>& result) {
    if (curr) {
        inorder<T>(curr->smaller_, result);

        result.push_back(curr);

        inorder<T>(curr->larger_, result);
    }
}

template <typename T>
void rnn_search(GKDTreeNode<T>* curr, T* data, std::vector<T*>& result_set, typename T::DistType radius) {
    if (curr->has_leaves()) {
        for (T* child : curr->leaves_) {
            if (T::is_in_radius(data, child, radius * radius)) {
                result_set.push_back(child);
            }
        }
    } else {
        if (curr->in_smaller_side(data)) {
            rnn_search<T>(curr->smaller_, data, result_set, radius);

            if (T::is_in_radius(data, curr->data_, curr->axis_, radius)) {
                rnn_search<T>(curr->larger_, data, result_set, radius);
            }
        } else if (curr->in_larger_side(data)) {
            rnn_search<T>(curr->larger_, data, result_set, radius);
            if (T::is_in_radius(data, curr->data_, curr->axis_, radius)) {
                rnn_search<T>(curr->smaller_, data, result_set, radius);
            }
        } else {
            rnn_search<T>(curr->smaller_, data, result_set, radius);
            rnn_search<T>(curr->larger_, data, result_set, radius);
        }
    }
}

template <typename T>
std::vector<T*> GKdTree<T>::rnn_search(T* data, typename T::DistType radius) const {
    std::vector<T*> result_set;
    ::rnn_search<T>(root_, data, result_set, radius);
    return result_set;
}


template <typename T>
GKdTree<T>::~GKdTree() {
    std::vector<PtrNode> ptr_nodes;
    ptr_nodes.reserve(size_);
    ::inorder(root_, ptr_nodes);

    for (auto ptr_node : ptr_nodes) {
        delete ptr_node;
        ptr_node = nullptr;
    }
}
~~~

~~~
Finish building KDTree, which costs 91.734milliseconds
Starting search data from the middle 
@@@@@@@@@RNN Search costs 0.006miliseconds
@@@@@@@@@Brute force costs 3.478miliseconds
~~~

