#include "fpfh.h"
#include "ransac.h"
#include "spfh.h"
#include <iostream>
int main(int argc, char **argv) {

  //   // resd ply data
  //   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new
  //   pcl::PointCloud<pcl::PointXYZ>); pcl::io::loadPLYFile(argv[1], *cloud);

  //   // define kdtree
  //   pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
  //       new pcl::search::KdTree<pcl::PointXYZ>());

  //   double radius = 0.02;

  //   pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_key_point_detector;
  //   // set iss parameters
  //   iss_key_point_detector.setSearchMethod(tree);
  //   iss_key_point_detector.setSalientRadius(20 * radius);
  //   iss_key_point_detector.setNonMaxRadius(10 * radius);
  //   iss_key_point_detector.setThreshold21(0.9);
  //   iss_key_point_detector.setThreshold32(0.9);
  //   iss_key_point_detector.setMinNeighbors(30);
  //   iss_key_point_detector.setNumberOfThreads(10);
  //   iss_key_point_detector.setInputCloud(cloud);

  //   pcl::PointCloud<pcl::PointXYZ>::Ptr keys(new
  //   pcl::PointCloud<pcl::PointXYZ>); iss_key_point_detector.compute(*keys);

  //   std::cout << "key points size : " << keys->size() << std::endl;

  //   // vis iss
  //   pcl::visualization::PCLVisualizer::Ptr viewer(
  //       new pcl::visualization::PCLVisualizer("Viewer"));
  //   viewer->setBackgroundColor(0, 0, 0);
  //   viewer->addPointCloud(cloud, "original_cloud");
  //   viewer->addPointCloud(keys, "keys");

  //   viewer->setPointCloudRenderingProperties(
  //       pcl::visualization::PCL_VISUALIZER_COLOR, 0., 1., 0.,
  //       "original_cloud");
  //   viewer->setPointCloudRenderingProperties(
  //       pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7.5, "keys");
  //   viewer->setPointCloudRenderingProperties(
  //       pcl::visualization::PCL_VISUALIZER_COLOR, 1., 0., 0., "keys");

  //   while (!viewer->wasStopped()) {
  //     viewer->spinOnce();
  //   }

  //   pcl::PointCloud<pcl::Normal>::Ptr normals(new
  //   pcl::PointCloud<pcl::Normal>());

  //   pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal>
  //   cloud_normal_estimator; cloud_normal_estimator.setKSearch(10);
  //   cloud_normal_estimator.setSearchSurface(cloud);
  //   cloud_normal_estimator.setInputCloud(cloud);
  //   cloud_normal_estimator.compute(*normals);

  //   // create fpfh estimator, pass the cloud & normals
  //   FPFHEstimator fpfh_estimator;
  //   fpfh_estimator.set_input_cloud(keys);
  //   fpfh_estimator.set_input_normal(normals);
  //   fpfh_estimator.set_search_surface(cloud);

  //   std::vector<FPFHSignature33> fpfh_descriptors;

  //   fpfh_estimator.set_radius_search(0.18);
  //   fpfh_estimator.compute(fpfh_descriptors);

  //   std::cout << "FPFH descriptor size: " << fpfh_descriptors.size() <<
  //   std::endl;

  return 0;
}