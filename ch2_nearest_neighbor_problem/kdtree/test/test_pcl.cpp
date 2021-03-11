
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

  pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_ptr(
      new pcl::PointCloud<pcl::PointXYZI>);

  read_point_cloud(path, point_cloud_ptr);

  pcl::visualization::CloudViewer viewer("Cloud viewer");

  while (!viewer.wasStopped()) {
    viewer.showCloud(point_cloud_ptr);
  }
  pcl::PointCloud<pcl::PointXYZI> point_cloud = *point_cloud_ptr;
  pcl::io::savePCDFileASCII("../data/test_pcd.pcd", point_cloud);
  std::cerr << "Saved " << point_cloud.size() << " data points to test_pcd.pcd."
            << std::endl;

  for (const auto &point : point_cloud)
    std::cerr << "    " << point.x << " " << point.y << " " << point.z << " "
              << point.intensity << std::endl;
  return (0);
}
