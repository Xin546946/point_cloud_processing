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
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPLYFile(argv[1], *cloud);

  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addPointCloud(cloud, "original_cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_COLOR, 0., 1., 0., "original_cloud");
  while (!viewer->wasStopped()) {
    viewer->spinOnce();
  }
  return 0;
}