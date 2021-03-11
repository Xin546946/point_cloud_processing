#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

void readbinfile(const std::string &path, pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud){
    std::fstream inputFile_(path, std::ios::in | std::ios::binary);
    while(inputFile_){
        pcl::PointXYZI point;
        inputFile_.read((char *) &point.x, 3*sizeof(float));
        //inputFile_.read((char *) &point.y, sizeof(float));
        //inputFile_.read((char *) &point.z, sizeof(float));
        inputFile_.read((char *) &point.intensity, sizeof(float));
        point_cloud->push_back(point);
    }
    inputFile_.close();
    return;      
}

int main(int argc, char **argv){
    std::string path = argv[1];//"/home/gfeng/gfeng_ws/point_cloud_processing/ch2_nearest_neighbor_problem/data/000000.bin";
    std::string outputFile_ = "/home/gfeng/gfeng_ws/point_cloud_processing/ch2_nearest_neighbor_problem/data/0.pcd";
    pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    readbinfile(path, point_cloud);
 
    /*pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    while(!viewer.wasStopped()){
        viewer.showCloud(point_cloud);
    }*/

    pcl::io::savePCDFileBinary(outputFile_, *point_cloud);
    if(pcl::io::loadPCDFile<pcl::PointXYZI>("/home/gfeng/gfeng_ws/point_cloud_processing/ch2_nearest_neighbor_problem/data/0.pcd", *point_cloud) == -1){
        PCL_ERROR ("Couldn't read file\n");
        return (-1);
    }

    //visualization
    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    viewer.showCloud(point_cloud);
    while (!viewer.wasStopped())
    {
    }
    return 0;
}