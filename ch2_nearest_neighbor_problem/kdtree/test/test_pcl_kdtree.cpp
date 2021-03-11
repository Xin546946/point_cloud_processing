#include "Eigen/Core"
#include "gkdtree.h"
#include "math_utils.h"
#include "tictoc.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/common/geometry.h>
#include <pcl/console/print.h>
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
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  std::string pcd_path = argv[1];
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *cloud) ==
      -1) { // load the point cloud
    PCL_ERROR("Could not read file test_pcd.pcd. \n");
    return -1;
  } else {
    std::cout << "Loaded " << cloud->width * cloud->height
              << " data points from test_pcd.pcd. " << std::endl;
  }
  // for (size_t i = 0; i < cloud->points.size(); ++i)
  //   std::cout << "    " << cloud->points[i].x << " " << cloud->points[i].y
  //             << " " << cloud->points[i].z << std::endl;
  std::vector<PointNode> point_node_vec(10000);
  for (size_t idx = 0; idx < cloud->points.size(); idx++) {
    point_node_vec.emplace_back(cloud->points[idx].x, cloud->points[idx].y,
                                cloud->points[idx].z);
  }

  std::cout << "Start build kdtree......\n";
  tictoc::tic();
  GKdTree<PointNode> gkdtree(&point_node_vec[0], point_node_vec.size());

  PointNode search_data(cloud->points[0].x, cloud->points[0].y,
                        cloud->points[0].z);
  tictoc::tic();
  std::vector<PointNode *> result = gkdtree.knn_search(&search_data, 8);
  std::cout << "8NN Search costs:  " << tictoc::toc() / 1e3 << "ms" << '\n';

  for (auto &node : result) {
    std::cout << node->point_ << '\n';
  }

  pcl::PointXYZ search_data_pcl(cloud->points[0].x, cloud->points[0].y,
                                cloud->points[0].z);

  tictoc::tic();
  std::vector<float> distances(10000);
  for (std::size_t idx = 0; idx < cloud->points.size(); idx++) {
    std::cout << cloud->points[idx] << '\n';
    float distance =
        pcl::geometry::squaredDistance(cloud->points[idx], search_data_pcl);
    distances.push_back(distance);
  }

  std::vector<float> test_data{0.1f, 0.3f, 0.2f, 0.4f};
  std::vector<size_t> sorted_idx = sort_indexes(test_data);

  for (std::size_t i = 0; i < sorted_idx.size(); i++) {
    std::cout << sorted_idx[i] << '\n';
  }

  std::vector<pcl::PointXYZ> result_set(8);

  for (std::size_t idx = 0; idx < 8; idx++) {
    result_set.push_back(cloud->points[sorted_idx[idx]]);
    std::cout << sorted_idx[idx] << " " << '\n';
  }
  std::cout << "Brute force costs: " << tictoc::toc() / 1e3 << "miliseconds"
            << '\n';
}