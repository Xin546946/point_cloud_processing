#include "iss.h"
#include "omp.h"
#include <algorithm>
#include <numeric>
#include <vector>

template <typename T> std::vector<int> sort_indexes(const std::vector<T> &vec) {
  std::vector<int> index(vec.size());
  std::iota(index.begin(), index.end(), 0);
  std::stable_sort(index.begin(), index.end(),
                   [&vec](int lhs, int rhs) { return vec[lhs] > vec[rhs]; });
  return index;
}

void ISSKeypoints::use_weighted_conv_matrix(
    bool should_use_weighted_conv_matrix) {
  this->use_weighted_conv_matrix_ = should_use_weighted_conv_matrix;
}

void ISSKeypoints::set_input_cloud(
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud) {
  this->point_cloud_ = point_cloud;
}

void ISSKeypoints::set_local_radius(float radius) {
  this->rnn_radius_ = radius;
}

void ISSKeypoints::set_non_max_radius(float radius) {
  this->non_max_radius_ = radius;
}

void ISSKeypoints::set_threshold(float g21, float g32) {
  this->gamma21_ = g21;
  this->gamma32_ = g32;
}

void ISSKeypoints::set_min_neighbors(int min_neighbor) {
  this->min_neighbors_ = min_neighbor;
}
void ISSKeypoints::compute(pcl::PointCloud<pcl::PointXYZ>::Ptr key_points) {

  // build kdtree
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(this->point_cloud_);
  int num_points = this->point_cloud_->size();

  // compute rnn indices of all points
  std::vector<std::vector<int>> rnn_indices(num_points);
  omp_set_num_threads(8);

#pragma omp parallel for
  for (int i = 0; i < num_points; i++) {
    // std::cout << this->point_cloud_->points[i] << '\n';
    pcl::PointXYZ search_point = this->point_cloud_->points[i];
    std::vector<float> distances;
    std::vector<int> rnn_idx;
    kdtree.radiusSearch(search_point, this->rnn_radius_, rnn_idx, distances);
    rnn_indices[i] = rnn_idx;
  }

  std::vector<float> lamda3_vec(num_points, -1);

#pragma omp parallel for
  // compute covariance matrix for each point
  for (int i = 0; i < num_points; i++) {
    // std::cout << "Process the " << i << "-th point" << '\n';
    int num_neighbors = rnn_indices[i].size();

    // compute covariance matrix for each point
    if (num_neighbors > this->min_neighbors_) {
      Eigen::Vector3f center_point{this->point_cloud_->points[i].x,
                                   this->point_cloud_->points[i].y,
                                   this->point_cloud_->points[i].z};

      Eigen::Matrix3f cov_matrix = Eigen::Matrix3f::Zero(3, 3);
      if (this->use_weighted_conv_matrix_) {
        float weight_sum = 0.f;
        float weight;
        for (int j = 0; j < num_neighbors; j++) {
          int idx_neighbors = rnn_indices[i][j];
          weight = 1.f / rnn_indices[idx_neighbors].size();
          Eigen::Vector3f neighbor_point{
              (*this->point_cloud_)[idx_neighbors].x,
              (*this->point_cloud_)[idx_neighbors].y,
              (*this->point_cloud_)[idx_neighbors].z};

          cov_matrix += weight * (neighbor_point - center_point) *
                        (neighbor_point - center_point).transpose();
          weight_sum += weight;
        }
        cov_matrix /= weight_sum;
        // std::cout << "Covariance matrix" << '\n';
        // std::cout << cov_matrix << '\n';
        // std::cout << '\n';
      }

      else {
        float weight_sum = 0.f;
        for (int j = 0; j < num_neighbors; j++) {
          int idx_neighbors = rnn_indices[i][j];

          Eigen::Vector3f neighbor_point{
              (*this->point_cloud_)[idx_neighbors].x,
              (*this->point_cloud_)[idx_neighbors].y,
              (*this->point_cloud_)[idx_neighbors].z};

          cov_matrix += (neighbor_point - center_point) *
                        (neighbor_point - center_point).transpose();
        }
        cov_matrix /= num_neighbors;
      }

      // compute eigenvalues, tipps: this eigenvalue is lamda1 < lamda2 < lamda3
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(cov_matrix);
      Eigen::Vector3f eigenvalues = eigensolver.eigenvalues().real();
      float lamda1 = eigenvalues[2];
      float lamda2 = eigenvalues[1];
      float lamda3 = eigenvalues[0];
      // std::cout << lamda1 << s" " << lamda2 << " " << lamda3 << '\n';
      if (lamda2 / lamda1 < this->gamma21_ &&
          lamda3 / lamda2 < this->gamma32_ && lamda3 > 0) {
        // std::cout << "This point is key spoint" << '\n';
        lamda3_vec[i] = lamda3;
      }
    }
  }

  std::vector<int> indices = sort_indexes(lamda3_vec);
  // //s std::vector<int> new_indices(indices.begin() + position,
  // indices.end());

  // indices.erase(indices.begin() + position, indices.end());
  // std::cout << "Non -1 indices remains: " << indices.size() << std::endl;
  // int counter =s 0;
  // int count_whisle = 0;
  // std::cout << "The number of good index are: " << indices.size() << '\n';
  while (!indices.empty()) {
    // count_while++;
    pcl::PointXYZ search_point = this->point_cloud_->points[indices[0]];
    // std::cout << "Now is biggest lamda: " << lamda3_vec[indices[0]] <<
    // '\n';
    std::vector<float> distances;
    std::vector<int> rnn_idx;
    kdtree.radiusSearch(search_point, this->non_max_radius_, rnn_idx,
                        distances);

    // std::cout << indices.empty() << '\n';
    bool is_key_point = true;
    for (int idx : rnn_idx) {
      // Here should tell if the lamda3_vec[max_indicex_idx] bigger than RNN
      // points
      if (lamda3_vec[idx] > lamda3_vec[indices[0]]) {
        is_key_point = false;
      }
      std::vector<int>::iterator it = find(indices.begin(), indices.end(), idx);

      if (it == indices.end())
        continue;
      indices.erase(it);
      // counter++;
    }
    if (is_key_point) {
      key_points->push_back(search_point);
    }
    // std::cout << "After one rnn deletion, it remains: " << indices.size()
    // << "indices."
    // << "RNN points: " << rnn_idx.size() << '\n';
  }
  // std::cout << "Delete " << counter << "points" << '\n';
  // std::cout << "NUm of while counter: " << count_while << '\n';
  // std::cout << "Key points size: " << key_points->size() << '\n';
  // #pragma omp parallel for
  // apply non-max_suppression
  // for (int i = 0; i < num_points; i++) {
  //   if (lamda3_vec[i] == -1) {
  //     continue;
  //   }
  //   bool is_key_point = true;
  //   pcl::PointXYZ search_point = this->point_cloud_->points[i];
  //   std::vector<float> distances;
  //   std::vector<int> rnn_idx;
  //   kdtree.radiusSearch(search_point, this->non_max_radius_, rnn_idx,
  //                       distances);

  //   if (rnn_idx.size() < this->min_neighbors_) {
  //     continue;
  //   }

  //   for (const auto &dist_idx : rnn_idx) {
  //     if (lamda3_vec[i] < lamda3_vec[dist_idx]) {
  //       is_key_point = false;
  //       break;
  //     }
  //   }

  //   if (is_key_point) {
  //     // std::cout << "Find a key points points[" << i << "]" << '\n';
  //     key_points->push_back(this->point_cloud_->points[i]);
  //     // std::cout << "Key point: " s<< this->point_cloud_->points[i] <<
  //     '\n';
  //   }
  // }
  // std::cout << "Key points: " << key_points->size() << std::endl;
}