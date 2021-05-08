#include "fpfh.h"
#include <iostream>
#include <vector>

class FPFHResultset {
public:
  FPFHResultset() = default;
  void add_histogran(
      std::vector<float> histogram,
      float weight); // compute histogram, set alpha_hist->phi_hist->theta_hist
  std::vector<FPFHSignature33> get_histogram();

private:
  std::vector<float> histogram_(33, 0.0f);
}

int main(int argc, char **argv) {
  std::vector<> histogram
}