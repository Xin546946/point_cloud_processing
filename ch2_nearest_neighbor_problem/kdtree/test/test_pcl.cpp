#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int main(int argc, char **argv) {
  std::string path = argv[0];
  std::vector<Eigen::Vector3f> points;
  points.clear();
  points.reserve(100000);
  std::ifstream file(path, std::ios::binary);
  float x, y, z, i;
  while (file) {
    file.read((char *)&x, sizeof(float));
    file.read((char *)&y, sizeof(float));
    file.read((char *)&z, sizeof(float));
    file.read((char *)&i, sizeof(float));
    points.emplace_back(x, y, z);
  }

  pcl::PointCloud<pcl::PointXYZI> cloud;

  for (int i = 0; i < points.size(); i++) {
    std::cout << points[i](0) << " " << points[i](1) << " " << points[i](2)
              << '\n';
  }

  pcl::io::savePCDFileASCII("test_pcd.pcd", cloud);
  std::cerr << "Saved " << cloud.size() << " data points to test_pcd.pcd."
            << std::endl;

  for (const auto &point : cloud)
    std::cerr << "    " << point.x << " " << point.y << " " << point.z
              << std::endl;

  return (0);
}
