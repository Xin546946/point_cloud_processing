#include "Eigen/Core"
#include "gkdtree.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

struct PointNode {
  PointNode() = default;
  PointNode(Eigen::Vector3f point) : point_(point) {}
  PointNode(float x, float y, float z) : point_(x, y, z) {}
  typedef int DistType;
  const uchar operator[](int idx) const { return this->point_(idx); }
  static bool is_in_radius(const PointNode *center, const PointNode *data,
                           int r_square) {
    return (center->point_ - data->point_).squaredNorm() < r_square;
  }
  static bool is_in_radius(const PointNode *center, const PointNode *data,
                           int axis, int radius) {
    return std::abs(center->point_(axis) - data->point_(axis)) < radius;
  }
  bool is_convergent();
  Eigen::Vector3f point_;
  static const int dim_ = 3;
};

int main(int argc, char **argv) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZI>);

  std::string pcd_path = argv[1];
  if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_path, *cloud) ==
      -1) { // load the point cloud
    PCL_ERROR("Could not read file test_pcd.pcd. \n");
    return -1;
  } else {
    std::cout << "Loaded " << cloud->width * cloud->height
              << " data points from test_pcd.pcd. " << std::endl;
  }
  // for (size_t i = 0; i < cloud->points.size(); ++i)
  // std::cout << "    " << cloud->points[i].x << " " << cloud->points[i].y
  // << " " << cloud->points[i].z << std::endl;

  return 0;
}