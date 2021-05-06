// #include "fpfh.h"
#include "spfh.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  std::vector<float> test_vec;
  for (int i = 1; i < 12; i++) {
    test_vec.push_back(float(i));
  }
  std::vector<float> test_vec2;
  for (int i = 1; i < 12; i++) {
    test_vec2.push_back(float(i));
  }
  std::vector<float> test_vec3;
  for (int i = 1; i < 12; i++) {
    test_vec3.push_back(float(i));
  }

  SPFHResultset spfh_resultset;
  spfh_resultset.set_triplet_features(test_vec, test_vec2, test_vec3);
  std::vector<float> hist = spfh_resultset.get_histogram();
  for (auto h : hist) {
    std::cout << h << ' ';
  }
  std::cout << std::endl;
}
