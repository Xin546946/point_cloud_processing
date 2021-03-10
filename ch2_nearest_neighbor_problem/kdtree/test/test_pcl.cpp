#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

void read_point_cloud(const std::string &path,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr points) {

  points->clear();
  points->reserve(100000);
  std::ifstream file(path, std::ios::binary);
  float x, y, z, i;
  while (file) {
    file.read((char *)&x, sizeof(float));
    file.read((char *)&y, sizeof(float));
    file.read((char *)&z, sizeof(float));
    file.read((char *)&i, sizeof(float));
    points->emplace_back(x, y, z, i);
  }
}

int main(int argc, char **argv) {
  std::string path = "/home/kit/point_cloud_processing/"
                     "ch2_nearest_neighbor_problem/kdtree/data/000000.bin";

  pcl::PointCloud<pcl::PointXYZI>::Ptr points_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);

  read_point_cloud(path, points_cloud);

  pcl::visualization::CloudViewer viewer("Cloud viewer");

  while (!viewer.wasStopped()) {
    viewer.showCloud(points_cloud);
  }
  pcl::PointCloud<pcl::PointXYZI> cloud = *points_cloud;
  pcl::io::savePCDFileASCII("../data/test_pcd.pcd", cloud);
  std::cerr << "Saved " << cloud.size() << " data points to test_pcd.pcd."
            << std::endl;

  // for (const auto &point : cloud)
  //   std::cerr << "    " << point.x << " " << point.y << " " << point.z
  //             << std::endl;
  return (0);
}
