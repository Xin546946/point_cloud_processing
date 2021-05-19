#include "ransac.h"
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  std::vector<std::vector<float>> pointset1 = {
      {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {3.0, 3.0, 3.0}};
  std::vector<std::vector<float>> pointset2 = {
      {3.0, 3.0, 3.0}, {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}};

  std::vector<int> correspondences_idx;

  find_correspondence(pointset1, pointset2, correspondences_idx);
  for (auto idx : correspondences_idx) {
    std::cout << idx << std::endl;
  }
}