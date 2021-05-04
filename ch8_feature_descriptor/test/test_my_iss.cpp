// #include "iss.cpp"
#include "fpfh.h"
#include "iss.h"
#include <chrono>
#include <iostream>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<float, std::milli> Duration;
typedef std::vector<std::vector<float>> PCDType;

int main(int argc, char **argv) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPLYFile(argv[1], *point_cloud);
  std::cout << "Point cloud width x height: " << point_cloud->width << " x "
            << point_cloud->height << '\n';

  size_t num_points = point_cloud->size();
  std::cout << "The number of points: " << num_points << '\n';

  float radius = 0.02;
  ISSKeypoints iss_detector;
  iss_detector.use_weighted_conv_matrix(true);
  iss_detector.set_input_cloud(point_cloud);
  iss_detector.set_local_radius(6 * radius);
  iss_detector.set_non_max_radius(4 * radius);
  iss_detector.set_threshold(0.9, 0.9);
  iss_detector.set_min_neighbors(5);

  pcl::PointCloud<pcl::PointXYZ>::Ptr keys(new pcl::PointCloud<pcl::PointXYZ>);
  auto start = Clock::now();

  iss_detector.compute(keys);
  //   std::cout << *keys << '\n';
  auto end = Clock::now();
  Duration duration = end - start;
  std::cout << "detection takes " << duration.count() << " ms" << std::endl;
  std::cout << "Key points size: " << keys->size() << '\n';

  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addPointCloud(point_cloud, "input");
  viewer->addPointCloud(keys, "keys");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, 0., 1., 0., "input");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7.5, "keys");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, 1., 0., 0., "keys");

  //   while (!viewer->wasStopped()) {
  //     viewer->spinOnce();
  //   }

  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());

  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> cloud_normal_estimator;
  cloud_normal_estimator.setKSearch(10);
  cloud_normal_estimator.setSearchSurface(point_cloud);
  cloud_normal_estimator.setInputCloud(point_cloud);
  cloud_normal_estimator.compute(*normals);

  // create fpfh estimator, pass the cloud & normals
  FPFHEstimator fpfh_estimator;
  fpfh_estimator.set_input_cloud(keys);
  fpfh_estimator.set_input_normal(normals);
  fpfh_estimator.set_search_surface(point_cloud);

  // declare descriptor

  fpfh_estimator.set_radius_search(0.18);
  std::vector<FPFHSignature33> fpfh_descriptors;
  fpfh_estimator.compute(fpfh_descriptors);

  std::cout << "FPFH descriptor size: " << fpfh_descriptors.size() << std::endl;

  //   for (int i = 0; i < 33; i++) {
  // std::cout << fpfh_descriptors->points[0].histogram[i] << '\n';
  //   }

  return 0;
}