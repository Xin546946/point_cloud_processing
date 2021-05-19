#pragma once
#include <vector>
#include "fpfh.h"
#include "spfh.h"
#include <pcl/point_types.h>

float compute_square_dist(FPFHSignature33 lhs, FPFHSignature33 rhs);

void find_correspondence(
    const std::vector<FPFHSignature33> &source_descriptor,
    const std::vector<FPFHSignature33> &target_descriptor,
    std::vector<int> &correspondences_idx);

void RANSAC(std::vector<std::pair<int,int>> correspondences_idx,
            pcl::PointCloud<pcl::PointXYZ>::Ptr source_key_points,
            pcl::PointCloud<pcl::PointXYZ>::Ptr target_key_points,
            Eigen::Matrix3f& R, 
            Eigen::Vector3f& t);