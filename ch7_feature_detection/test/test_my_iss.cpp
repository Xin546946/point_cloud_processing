// #include "iss.cpp"
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

typedef std::vector<std::vector<float>> PCDType;

int main(int argc, char **argv) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPLYFile(argv[1], *cloud);
  std::cout << "Point cloud width x height: " << cloud->width << " x "
            << cloud->height << '\n';

  size_t num_points = cloud->size();
  std::cout << "The number of points: " << num_points << '\n';
  PCDType point_cloud(num_points, std::vector<float>(3));

  for (int i = 0; i < num_points; i++) {
    point_cloud[i][0] = cloud->points[i].x;
    point_cloud[i][1] = cloud->points[i].y;
    point_cloud[i][2] = cloud->points[i].z;
  }

  float radius = 0.02;
  ISSKeypoints iss_detector;
  iss_detector.use_weighted_conv_matrix(true);
  iss_detector.set_input_cloud(point_cloud);
  iss_detector.set_local_radius(6 * radius);
  iss_detector.set_non_max_radius(4 * radius);
  iss_detector.set_threshold(0.9, 0.9);
  iss_detector.set_min_neighbors(5);

  PCDType key_points;
  iss_detector.compute(key_points);

  std::cout << "Key points size: " << key_points.size() << '\n';

  pcl::PointCloud<pcl::PointXYZ>::Ptr keys(new pcl::PointCloud<pcl::PointXYZ>);
  for (const auto &key_point : key_points) {
    keys->points.emplace_back(
        pcl::PointXYZ(key_point[0], key_point[1], key_point[2]));

    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud(cloud, "input");
    viewer->addPointCloud(keys, "keys");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, 0., 1., 0., "input");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7.5, "keys");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, 1., 0., 0., "keys");

    while (!viewer->wasStopped()) {
      viewer->spinOnce();
    }
  }

  return 0;
}