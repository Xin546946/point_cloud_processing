#include <chrono>
#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>

int main(int argc, char **argv) {
  // resd ply data
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPLYFile(argv[1], *cloud);

  // define kdtree
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
      new pcl::search::KdTree<pcl::PointXYZ>());

  double radius = 0.02;

  pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_key_point_detector;
  // set iss parameters
  iss_key_point_detector.setSearchMethod(tree);
  iss_key_point_detector.setSalientRadius(6 * radius);
  iss_key_point_detector.setNonMaxRadius(4 * radius);
  iss_key_point_detector.setThreshold21(0.9);
  iss_key_point_detector.setThreshold32(0.9);
  iss_key_point_detector.setMinNeighbors(5);
  iss_key_point_detector.setNumberOfThreads(4);
  iss_key_point_detector.setInputCloud(cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr keys(new pcl::PointCloud<pcl::PointXYZ>);
  iss_key_point_detector.compute(*keys);

  std::cout << "key points size : " << keys->size() << std::endl;

  // vis iss
  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addPointCloud(cloud, "original_cloud");
  viewer->addPointCloud(keys, "keys");

  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, 0., 1., 0., "original_cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7.5, "keys");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, 1., 0., 0., "keys");

  while (!viewer->wasStopped()) {
    viewer->spinOnce();
  }

  return 0;
}