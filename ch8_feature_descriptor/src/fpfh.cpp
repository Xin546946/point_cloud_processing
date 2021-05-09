#include "fpfh.h"
#include "spfh.h"
#include <math.h>

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
/*--------------------------------------------------------
  #####################implementation: FPFHResult #####################
  ---------------------------------------------------------*/

void FPFHResultset::add_histogram(std::vector<float> histogram, float weight) {
  histogram = weight * histogram;
  this->histogram_ = this->histogram_ + histogram;
}

std::vector<float> FPFHResultset::get_histogram() { return this->histogram_; }

Eigen::Vector3f compute_triplet_feature(Eigen::Vector3f diff,
                                        Eigen::Vector3f normal1,
                                        Eigen::Vector3f normal2) {

  Eigen::Vector3f v = normal1.cross(diff / (diff.norm() + 1e-8));
  Eigen::Vector3f w = normal1.cross(v);

  float alpha = v.dot(normal2);
  float phi = normal1.dot(diff / (diff.norm() + 1e-8));
  float theta = atan(w.dot(normal2) / normal1.dot(normal2));

  return Eigen::Vector3f(alpha, phi, theta);
}

void FPFHEstimator::set_input_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr keys) {
  this->keys_ = keys;
}

void FPFHEstimator::set_input_normal(
    pcl::PointCloud<pcl::Normal>::Ptr normals) {
  this->normals_ = normals;
}

void FPFHEstimator::set_search_surface(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
  this->cloud_ = cloud;
}

void FPFHEstimator::set_radius_search(float radius) { this->radius_ = radius; }

void FPFHEstimator::compute(std::vector<FPFHSignature33> &fpfh_descriptor) {

  int num_keys = this->keys_->size();

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(this->cloud_);

  // float weight); get_histogram();

  for (int i = 0; i < num_keys; i++) {
    FPFHResultset
        fpfh_result_set; // add_histogram(std::vector<float> histogram,

    SPFHResultset key_spfh_resultset; // spfh for key point

    // compute rnn of key point
    std::vector<float> distances;
    std::vector<int> rnn_idx;
    kdtree.radiusSearch(this->keys_->points[i], this->radius_, rnn_idx,
                        distances);
    int num_rnn_point = rnn_idx.size() - 1;
    std::vector<float> alpha_vec, phi_vec, theta_vec;

    // compute rnn for key
    for (int idx : rnn_idx) {

      if (idx == rnn_idx[0]) {
        continue;
      }
      // compute spsh for key point
      Eigen::Vector3f key(this->keys_->points[i].x, this->keys_->points[i].y,
                          this->keys_->points[i].z);
      Eigen::Vector3f point(this->cloud_->points[idx].x,
                            this->cloud_->points[idx].y,
                            this->cloud_->points[idx].z);
      Eigen::Vector3f diff = point - key;

      // key normal is u
      Eigen::Vector3f key_normal(this->normals_->points[rnn_idx[0]].normal_x,
                                 this->normals_->points[rnn_idx[0]].normal_y,
                                 this->normals_->points[rnn_idx[0]].normal_z);

      Eigen::Vector3f point_normal(this->normals_->points[idx].normal_x,
                                   this->normals_->points[idx].normal_y,
                                   this->normals_->points[idx].normal_z);

      Eigen::Vector3f triplet_feature =
          compute_triplet_feature(diff, key_normal, point_normal);

      alpha_vec.push_back(triplet_feature[0]);
      phi_vec.push_back(triplet_feature[1]);
      theta_vec.push_back(triplet_feature[2]);

      // kdtree.radiusSearch(this->cloud_->points[idx], this->radius_,
      // sub_rnn_idx,
      //                     sub_distances);

      SPFHResultset neighbour_spfh_result_set; // spfh for neighbour

      std::vector<float> alpha_neighbour_vec, phi_neighbour_vec,
          theta_neighbour_vec;

      std::vector<float> sub_distances;
      std::vector<int> sub_rnn_idx;
      kdtree.radiusSearch(this->cloud_->points[idx], this->radius_, sub_rnn_idx,
                          sub_distances);
      float weight;
      // compute rnn for points[idx] , where we compute histogram on each
      // points[idx]
      for (int sub_idx : sub_rnn_idx) {
        // compute spsh for point[idx]
        if (sub_idx == idx) {
          continue;
        }
        Eigen::Vector3f point_neighbour(this->cloud_->points[sub_idx].x,
                                        this->cloud_->points[sub_idx].y,
                                        this->cloud_->points[sub_idx].z);
        Eigen::Vector3f diff_neighbour = point_neighbour - point;

        Eigen::Vector3f point_neighbour_normal(
            this->normals_->points[sub_idx].normal_x,
            this->normals_->points[sub_idx].normal_y,
            this->normals_->points[sub_idx].normal_z);

        Eigen::Vector3f triplet_feature_neighbour = compute_triplet_feature(
            diff_neighbour, point_normal, point_neighbour_normal);

        alpha_neighbour_vec.push_back(triplet_feature_neighbour[0]);
        phi_neighbour_vec.push_back(triplet_feature_neighbour[1]);
        theta_neighbour_vec.push_back(triplet_feature_neighbour[2]);
      }
      // if (diff.norm() == 0.f) {
      //   continue;
      // }
      weight = 1.f / (diff.norm() * num_rnn_point + 1e-8);
      neighbour_spfh_result_set.set_triplet_features(
          alpha_neighbour_vec, phi_neighbour_vec, theta_neighbour_vec);
      fpfh_result_set.add_histogram(neighbour_spfh_result_set.get_histogram(),
                                    weight);
    }

    key_spfh_resultset.set_triplet_features(alpha_vec, phi_vec, theta_vec);

    fpfh_result_set.add_histogram(key_spfh_resultset.get_histogram(), 1.f);
    fpfh_descriptor.push_back(fpfh_result_set.get_histogram());
  }
}
