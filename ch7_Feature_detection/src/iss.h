#pragma once
#include <cstdint>
#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/kdtree/kdtree_flann.h>

typedef pcl::PointCloud<pcl::PointXYZ>::Ptr CloudPtr;
class ISSKeypoint
{
public:
    void useWeightedCovMat(bool use);
    void setLocalRadius(float r);
    void setNonMaxRadius(float r);
    void setThreshold(float g21, float g32);
    void setMinNeighbors(int n);
    void setInputPointCloud(CloudPtr input_point_cloud);
    void compute(CloudPtr keypoints);
    
private:
    Eigen::Vector3f getEigenvalues(size_t i);

private:
    bool use_w_cov_mat;
    float local_radius;
    float non_max_radius;
    float gamma21, gamma32;
    int min_neighbors;
    CloudPtr point_cloud;
    std::vector<std::vector<int>> neighbors;
};