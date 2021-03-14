
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

void octree_knn_search(pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud, int K,
                       std::vector<int> &neighbor_idx,
                       std::vector<float> &neighbor_distances) {}

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