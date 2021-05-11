#pragma once
#include <iostream>
#include <vector>

class SPFHResultset{
    public:
    SPFHResultset() = default;
    void set_triplet_features(std::vector<float> alpha_vec, std::vector<float> phi_vec, std::vector<float> theta_vec);
    std::vector<float> get_histogram();

    private:
    std::vector<float> compute_histogram_from_vector(std::vector<float> vec);
    // int get_bin_id(float value, float width_bin);
    std::vector<float> histogram_;
};