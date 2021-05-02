#pragma once
#include <vector>
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
    void compute(std::vector<FPFHSignature33> fpfh_descriptor);

    private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr keys_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
    float radius_;
};

std::cout << fpfh_descriptors [i].histogram[i]