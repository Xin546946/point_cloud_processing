#include <iostream>
#include <random>
#include <vector>

std::vector<int> generate_random_data(int num, int min, int max) {
  std::random_device rd{};
  std::mt19937 gen{rd()};

  std::uniform_int_distribution<int> uniform_dist(min, max);
  std::vector<int> data;
  int id = uniform_dist(gen);
  data.push_back(id);

  while (1) {
    id = uniform_dist(gen);
    if (id != data.back()) {
      data.push_back(id);
      break;
    }
  }

  while (1) {
    id = uniform_dist(gen);
    if (id != data[0] and id != data[1]) {
      data.push_back(id);
      break;
    }
  }

  return data;
}

int main(int argc, char **argv) {
  std::vector<int> data = generate_random_data(3, 0, 100);
  for (int d : data) {
    std::cout << d << '\n';
  }
  return 0;
}