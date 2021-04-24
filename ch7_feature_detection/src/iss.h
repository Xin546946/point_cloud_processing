#pragma once

#include <vector>
#include "Eigen/Eigenvalues"
class ISSKeypoints{
    public:
    ISSKeypoints() = default;
    void use_weighted_conv_matrix(bool should_use_weighted_conv_matrix);
    void set_input_cloud(std::vector<std::vector<float>>& point_cloud);
    void set_local_radius(float radius);
    void set_non_max_radius(float radius);
    void set_threshold(float g21, float g32);
    void set_min_neighbors(int min_neighbor);
    void compute(std::vector<std::vector<float>>& key_points);

    private:
    Eigen::Vector3f get_eigen_values(size_t i);

    bool use_weighted_conv_matrix_;
    float local_radius_;
    float non_max_radius_;
    float gama21_;
    float gama32_;
    int min_neighbors_;
    std::vector<std::vector<int>> rnn_idx_;
    std::vector<std::vector<float>> point_cloud_;

};