#pragma once

#include <vector>
#include "Eigen/Eigenvalues"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
class ISSKeypoints{
    public:
    ISSKeypoints() = default;
    void use_weighted_conv_matrix(bool should_use_weighted_conv_matrix);
    void set_input_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud);
    void set_local_radius(float radius);
    void set_non_max_radius(float radius);
    void set_threshold(float g21, float g32);
    void set_min_neighbors(int min_neighbor);
    void compute(pcl::PointCloud<pcl::PointXYZ> key_points);

    private:
    Eigen::Vector3f get_eigen_values(size_t i);

    bool use_weighted_conv_matrix_;
    float rnn_radius_;
    float non_max_radius_;
    float gamma21_;
    float gamma32_;
    int min_neighbors_;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_;

};