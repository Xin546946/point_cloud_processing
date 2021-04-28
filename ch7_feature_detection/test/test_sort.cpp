#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
  std::vector<float> test_vec{2.f, 0.4, 5.4, 6.7, 34.2, 2.1, 1.1};
  // sort and change the order of vector in increasing order
  //   std::sort(test_vec.begin(), test_vec.end());
  std::vector<size_t> index(test_vec.size());
  std::iota(index.begin(), index.end(), 0);
  std::stable_sort(index.begin(), index.end(),
                   [&test_vec](size_t lhs, size_t rhs) {
                     return test_vec[lhs] > test_vec[rhs];
                   });

  //   for (auto elem : index) {
  //     std::cout << elem << '\n';
  //   }

  test_vec.erase(test_vec.begin() + 1);
  for (auto elem : test_vec) {
    std::cout << elem << '\n';
  }

  test_vec.erase(test_vec.begin() + 1);
  for (auto elem : test_vec) {
    std::cout << elem << '\n';
  }

  return 0;
}
