#include "ransac.h"
#include "fpfh.h"
#include "spfh.h"
#include <cmath>
#include <math.h>
#include <numeric>
#include <random>
#include <vector>
// #include <pcl/features/fpfh.h>
// #include <pcl/features/fpfh_omp.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/features/normal_3d_omp.h>
// #include <pcl/io/ply_io.h>
// #include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/keypoints/iss_3d.h>
// #include <pcl/search/impl/search.hpp>
// #include <pcl/search/kdtree.h>
// #include <pcl/visualization/pcl_visualizer.h>

std::vector<int> generate_3_random_numbers(int num, int min, int max) {
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

float compute_square_dist(FPFHSignature33 lhs, FPFHSignature33 rhs) {
  float result = 0.f;

  for (int i = 0; i < lhs.size(); i++) {

    result += std::pow(lhs[i] - rhs[i], 2);
  }

  return result;
}

void find_correspondence(const std::vector<FPFHSignature33> &source_descriptor,
                         const std::vector<FPFHSignature33> &target_descriptor,
                         std::vector<int> &correspondences_idx) {
  // make sure source descriptor and target descriptor has the same size
  assert(source_descriptor.size() == target_descriptor.size());
  int correspondence_candidate;
  for (int i = 0; i < source_descriptor.size(); i++) {
    float min_dist_source_to_target = std::numeric_limits<float>::max();
    for (int j = 0; j < source_descriptor.size(); j++) {
      float source_to_target_dist =
          compute_square_dist(source_descriptor[i], target_descriptor[j]);
      if (source_to_target_dist < min_dist_source_to_target) {
        correspondence_candidate = j;
        min_dist_source_to_target = source_to_target_dist;
      }
    }
    correspondences_idx.push_back(correspondence_candidate);
  }
}

void RANSAC(std::vector<int> correspondences_idx,
            pcl::PointCloud<pcl::PointXYZ>::Ptr source_key_points,
            pcl::PointCloud<pcl::PointXYZ>::Ptr target_key_points,
            Eigen::Matrix3f &R, Eigen::Vector3f &t) {
  std::vector<int> random_indices =
      generate_3_random_numbers(3, 0, correspondences_idx.size() - 1);
  // define A,B matrix
  Eigen::Matrix3f A << source_key_points->points[random_indices[0]].x,
      source_key_points->points[random_indices[1]].x,
      source_key_points->points[random_indices[2]].x,
      source_key_points->points[random_indices[0]].y,
      source_key_points->points[random_indices[1]].y,
      source_key_points->points[random_indices[2]].y,
      source_key_points->points[random_indices[0]].z,
      source_key_points->points[random_indices[1]].z,
      source_key_points->points[random_indices[2]].z;

  Eigen::Matrix3f B
      << source_key_points->points[correspondences_idx[random_indices[0]]].x,
      source_key_points->points[correspondences_idx[random_indices[1]]].x,
      source_key_points->points[correspondences_idx[random_indices[2]]].x,
      source_key_points->points[correspondences_idx[random_indices[0]]].y,
      source_key_points->points[correspondences_idx[random_indices[1]]].y,
      source_key_points->points[correspondences_idx[random_indices[2]]].y,
      source_key_points->points[correspondences_idx[random_indices[0]]].z,
      source_key_points->points[correspondences_idx[random_indices[1]]].z,
      source_key_points->points[correspondences_idx[random_indices[2]]].z;
  // todo coplane
  // define L matrix to normalize the points
  Eigen::Matrix3f L =
      Eigen::Matrix3f::Identity(3, 3) - 1 / 3 * Eigen::Matrix3f::Ones();

  Eigen::Matrix3f A_norm = A * L;
  Eigen::Matrix3f B_norm = B * L;

  Eigen::JacobiSVD<Eigen::Matrix3f> svd(
      B_norm * A_norm.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f U, V, R;
  U = svd.matrixU();
  V = svd.matrixV();
  R = U * V.transpose();
  t = 1 / 3 * (B_norm - R * A_norm) * Eigen::Vector3f::Ones();
}
