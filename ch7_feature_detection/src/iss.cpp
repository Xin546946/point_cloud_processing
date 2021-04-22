#include "iss.h"

void ISSKeypoints::use_weighted_conv_matrix(
    bool should_use_weighted_conv_matrix) {
  this->use_weighted_conv_matrix_ = should_use_weighted_conv_matrix;
}

void ISSKeypoints::set_input_cloud(
    std::vector<std::vector<float>> &point_cloud) {
  this->point_cloud_ = point_cloud;
}

void ISSKeypoints::set_local_radius(float radius) {
  this->local_radius_ = radius;
}

void ISSKeypoints::set_non_max_radius(float radius) {
  this->non_max_radius_ = radius;
}

void ISSKeypoints::set_threshold(float g21, float g32) {
  this->gama21_ = g21;
  this->gama32_ = g32;
}

void ISSKeypoints::set_min_neighbors(int min_neighbor) {
  this->min_neighbors_ = min_neighbor;
}
void compute(std::vector<std::vector<float>> &key_points) {}