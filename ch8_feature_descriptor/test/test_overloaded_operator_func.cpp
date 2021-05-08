#include <iostream>
#include <vector>

std::vector<float> operator+(std::vector<float> &histogram1,
                             std::vector<float> &histogram2) {
  //   assert(histogram1.size() == histogram2.size());
  int num_elem = histogram1.size();
  std::vector<float> result(num_elem);
  for (int i = 0; i < histogram1.size(); i++) {
    result[i] = histogram1[i] + histogram2[i];
  }
  return result;
}

std::vector<float> operator*(float factor, std::vector<float> &vec) {
  std::vector<float> result(vec.size());
  for (int i = 0; i < vec.size(); i++) {
    result[i] = vec[i] * factor;
  }
  return result;
}

int main(int argc, char **argv) {
  std::vector<float> vec{1.0, 1.0, 1.0, 1.0};
  std::vector<float> test_plus_equal_operator(4, 10.f);
  std::vector<float> test_multiply_operator(4);
  //   vec *= 3.0;
  test_plus_equal_operator = vec + vec;
  test_multiply_operator = (5.f * vec) + (1.f * vec);

  for (int i = 0; i < vec.size(); i++) {
    std::cout << test_multiply_operator[i] << '\n';
  }
  return 0;
}