#include "spfh.h"
#include "fpfh.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

// std::vector<float> operator*(float factor, std::vector<float> &vec) {
//   std::vector<float> result(vec.size());
//   for (int i = 0; i < vec.size(); i++) {
//     result[i] = vec[i] * factor;
//   }
//   return result;
// }

void SPFHResultset::set_triplet_features(std::vector<float> alpha_vec,
                                         std::vector<float> phi_vec,
                                         std::vector<float> theta_vec) {
  std::vector<float> alpha_hist = compute_histogram_from_vector(alpha_vec);
  std::vector<float> phi_hist = compute_histogram_from_vector(phi_vec);
  std::vector<float> theta_hist = compute_histogram_from_vector(theta_vec);
  alpha_hist.insert(alpha_hist.end(), phi_hist.begin(), phi_hist.end());
  alpha_hist.insert(alpha_hist.end(), theta_hist.begin(), theta_hist.end());
  this->histogram_ = alpha_hist;
}

std::vector<float>
SPFHResultset::compute_histogram_from_vector(std::vector<float> vec) {
  std::vector<float> histogram(11);
  std::sort(vec.begin(), vec.end());
  float min_value = vec[0];
  float max_value = vec[vec.size() - 1];

  float width_bin = (max_value - min_value) / 11.f;
  for (float data : vec) {
    int bin_id = std::floor((data - min_value) / width_bin);
    histogram[bin_id]++;
  }
  // float sum = std::accumulate(histogram.begin(), histogram.end(), 0);
  // for (float &h : histogram) {
  //   h /= 11;
  // }
  return histogram;
}

std::vector<float> SPFHResultset::get_histogram() { return this->histogram_; }