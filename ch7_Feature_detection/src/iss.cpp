#include "iss.h"

void ISSKeypoint::useWeightedCovMat(bool use){
    this->use_w_cov_mat = use;
}

void ISSKeypoint::setLocalRadius(float r){
    this->local_radius = r;
}


void ISSKeypoint::setNonMaxRadius(float r){
    this->non_max_radius = r;
}

void ISSKeypoint::setThreshold(float g21, float g32){
    this->gamma21 = g21;
    this->gamma32 = g32;
}

void ISSKeypoint::setMinNeighbors(int n){
    this->min_neighbors = n;
}

void ISSKeypoint::setInputPointCloud(CloudPtr cld){
    this->point_cloud = cld;
    this->neighbors.resize(point_cloud->size());
}

void ISSKeypoint::compute(CloudPtr keypoints){

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(this->point_cloud);
    std::vector<float> potkps(point_cloud->size(), -1);
    for(int i = 0; i < this->point_cloud->size(); i++){
        // Neighbors within radius search
        std::vector<float> pointRadiusSquaredDistance;
        pcl::PointXYZ searchPoint = this->point_cloud->points[i];
        kdtree.radiusSearch(searchPoint, this->local_radius, neighbors[i], pointRadiusSquaredDistance);
    }

    for(int i = 0; i < point_cloud->size(); i++){
        std::cout<<"processing "<<i<<"th point"<<std::endl;
        if(neighbors[i].size() > min_neighbors){
            
        //create weighted cov
            pcl::PointXYZ cur_point = point_cloud->points[i];
            Eigen::Vector3f center_point{(*point_cloud)[i].x,
                                        (*point_cloud)[i].y,
                                        (*point_cloud)[i].z};
            Eigen::Matrix3f cov_matrix = Eigen::Matrix3f::Zero(3, 3);
            if(use_w_cov_mat){
                float weight;
                float weight_sum = 0;
                for(int j = 0; j < neighbors[i].size(); j++){
                    weight = 1.0f / neighbors[neighbors[i][j]].size();
                    if(weight == 0){
                        throw std::runtime_error("no neighbor");
                    }
                    weight_sum += weight;
                    Eigen::Vector3f neighbor{(*point_cloud)[neighbors[i][j]].x,
                                            (*point_cloud)[neighbors[i][j]].y,
                                            (*point_cloud)[neighbors[i][j]].z};
                    //std::cout<<neighbor<<std::endl;
                    cov_matrix += weight * (neighbor - center_point) * (neighbor - center_point).transpose();
                    //std::cout<<cov_matrix<<std::endl;
                }
                cov_matrix = cov_matrix / weight_sum;
                //std::cout<<cov_matrix<<std::endl;
            }
            else{
                for(int j = 0; j < neighbors[i].size(); j++){
                    Eigen::Vector3f neighbor{(*point_cloud)[neighbors[i][j]].x,
                                            (*point_cloud)[neighbors[i][j]].y,
                                            (*point_cloud)[neighbors[i][j]].z};
                    cov_matrix += (center_point - neighbor) * (center_point - neighbor).transpose();
                }
                cov_matrix = cov_matrix / neighbors[i].size();
            }
            //find eigenvalues
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(cov_matrix);
            Eigen::Vector3f eigenvalues = eigensolver.eigenvalues().real();
            //std::cout<<eigenvalues<<std::endl;
            if(eigenvalues[1] / eigenvalues[2] < gamma21 && eigenvalues[0] / eigenvalues[1] < gamma32 && eigenvalues[0] > 0){
                potkps[i] = eigenvalues[0];
            }
        }
    }
    //nonmax supression
    for (int i = 0; i < point_cloud->size(); i++)
    {
        if (potkps[i] == -1) continue;

        std::vector<float> pointRadiusSquaredDistance;
        std::vector<int> indices;
        pcl::PointXYZ searchPoint = this->point_cloud->points[i];
        kdtree.radiusSearch(searchPoint, this->non_max_radius, indices, pointRadiusSquaredDistance);
        if (indices.size() < min_neighbors) continue;

        bool is_keypoint = true;
        for (const int& dist_idx : indices)
            if (potkps[i] < potkps[dist_idx])
            {
                is_keypoint = false;
                break;
            }

        if (is_keypoint)
            keypoints->push_back(point_cloud->points[i]);

    }
}