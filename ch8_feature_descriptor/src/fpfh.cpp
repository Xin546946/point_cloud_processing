#include "fpfh.h"
#include <math.h>
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

void FPFHEstimator::compute(std::vector<FPFHSignature33> fpfh_descriptor) {
  int num_keys = this->keys_->size();
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(this->cloud_);

  Histogram histogram(11);

  for (int i = 0; i < num_keys; i++) {
    std::vector<float> distances;
    std::vector<int> rnn_idx;
    kdtree.radiusSearch(this->keys_->points[i], this->radius_, rnn_idx,
                        distances);
    // key normal is u
    Eigen::Vector3f key_normal(this->normals_->points[rnn_idx[0]].normal_x,
                               this->normals_->points[rnn_idx[0]].normal_y,
                               this->normals_->points[rnn_idx[0]].normal_z);
    std::vector<float> alpha_vec;
    std::vector<float> phi_vec;
    std::vector<float> theta_vec;
    for (int idx : rnn_idx) {

      Eigen::Vector3f diff(
          this->cloud_->points[idx].x - this->keys_->points[i].x,
          this->cloud_->points[idx].y - this->keys_->points[i].y,
          this->cloud_->points[idx].z - this->keys_->points[i].z);
      Eigen::Vector3f v = key_normal.cross(diff / diff.norm());
      Eigen::Vector3f w = key_normal.cross(v);

      // normal point is n2
      Eigen::Vector3f normal_point(this->normals_->points[idx].normal_x,
                                   this->normals_->points[idx].normal_y,
                                   this->normals_->points[idx].normal_z);
      float alpha = v.dot(normal_points);
      float phi = key_normal.dot(diff);
      float theta = atan2(w.dot(normal_point), key_normal.dot(normal_point));
      alpha_vec.push_back(alpha);
      phi_vec.push_back(phi);
      theta_vec.push_back(theta);

      std::vector<float> sub_distances;
      std::vector<int> sub_rnn_idx;
      kdtree.radiusSearch(this->cloud_->points[idx], this->radius_, sub_rnn_idx,
                          sub_distances);
    }
    histogram.set_input(alpha_vec);
    std::vector<float> hist_alpha = histogram.get_histogram();

    histogram.set_input(phi_vec);
    std::vector<float> hist_phi = histogram.get_histogram();

    histogram.set_input(theta_vec);
    std::vector<float> hist_theta = histogram.get_histogram();

    hist_alpha.insert(hist_alpha.end(), hist_phi.begin(), hist_phi.end());
    hist_alpha.insert(hist_alpha.end(), hist_theta.begin(), hist_theta.end());

    spfh_.push_back(hist_alpha);
  }
}