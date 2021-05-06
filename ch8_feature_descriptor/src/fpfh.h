#pragma once
#include <vector>
#include <Eigen/Dense>
#include "Eigen/Eigenvalues"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>

typedef std::vector<float> FPFHSignature33;

class FPFHEstimator{
    public:
    FPFHEstimator() = default;
    void set_input_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr keys);
    void set_input_normal(pcl::PointCloud<pcl::Normal>::Ptr normals);
    void set_search_surface(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    void set_radius_search(float radius);
    void compute(std::vector<FPFHSignature33>& fpfh_descriptor);

    private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr keys_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
    std::vector<FPFHSignature33> fpfh_;
    float radius_;
};

class FPFHResultset{
public: 
    FPFHResultset() = default;
    FPFHResultset(int num_bin):num_bin_(num_bin){}; 
    void add_histogran(std::vector<float> histogram); // compute histogram, set alpha_hist->phi_hist->theta_hist
    std::vector<FPFHSignature33> get_histogram();

    private:
    int num_bin_ = 11;
    std::vector<float> histogram_(num_bin);

} 
