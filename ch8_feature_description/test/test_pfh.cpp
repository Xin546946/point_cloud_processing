#include <pcl/features/fpfh.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <time.h>
#include <pcl/common/io.h>
#include <iostream>
#include <pcl/keypoints/iss_3d.h>//关键点检测

#include <pcl/search/impl/search.hpp>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/kdtree/kdtree_flann.h>

using pcl::NormalEstimation;
using pcl::search::KdTree;
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

//点云可视化
void visualize_pcd(PointCloud::Ptr pcd_src,
                   PointCloud::Ptr pcd_tgt)
    //PointCloud::Ptr pcd_final)
{
    pcl::visualization::PCLVisualizer viewer("registration Viewer");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_h(pcd_src, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tgt_h(pcd_tgt, 255, 0, 0);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> final_h(pcd_final, 0, 0, 255);
    viewer.setBackgroundColor(255, 255, 255);
    viewer.addPointCloud(pcd_src, src_h, "source cloud");
    viewer.addPointCloud(pcd_tgt, tgt_h, "tgt cloud");
    //viewer.addPointCloud(pcd_final, final_h, "final cloud");
    
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "tgt cloud");
    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    //pcl中sift特征需要返回强度信息，改为如下:
}

int main(int argc, char** argv)
{
    std::string path = argv[1];
    //加载点云文件
    PointCloud::Ptr input_cloud(new PointCloud);//原点云，待配准
    //"/home/gfeng/gfeng_ws/point_cloud_processing/ch7_Feature_detection/data/airplane.ply"
    if(pcl::io::loadPLYFile(path, *input_cloud) == -1){
        PCL_ERROR ("Couldn't read file\n");
        return (-1);
    }
    std::cout << "/////////////////////////////////////////////////" <<std::endl;
    std::cout << "原始点云数量："<<input_cloud->size() <<std::endl;

    PointCloud::Ptr keypointCloud(new PointCloud);
    //pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoint(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_det;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_1(new pcl::search::KdTree<pcl::PointXYZ>());

    double model_solution = 0.4;//参数小，采取的关键点多，论文中为500左右

    //参数设置
    iss_det.setSearchMethod(tree_1);
    iss_det.setSalientRadius(0.12);//
    iss_det.setNonMaxRadius(0.08);//
    iss_det.setThreshold21(0.975);
    iss_det.setThreshold32(0.975);
    iss_det.setMinNeighbors(5);
    iss_det.setNumberOfThreads(4);
    iss_det.setInputCloud(input_cloud);
    iss_det.compute(*keypointCloud);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> norm_est;
    norm_est.setKSearch(10);
    norm_est.setSearchSurface(input_cloud);
    norm_est.setInputCloud(input_cloud);
    norm_est.compute(*normals);

    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(keypointCloud);
    fpfh.setInputNormals(normals);
    fpfh.setSearchSurface(input_cloud);
    fpfh.setSearchMethod(tree_1);
    // Output datasets
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_descriptors (new pcl::PointCloud<pcl::FPFHSignature33>());

    fpfh.setRadiusSearch(0.18);
    fpfh.compute(*fpfh_descriptors);

    std::cout << "fpfh_descriptors size " << fpfh_descriptors->size() << std::endl;
    
	for ( size_t i = 0; i < 33 ; i++ )
	{
		std::cout << fpfh_descriptors->points[0].histogram[i] << std::endl;
	}
	

    //PointCloud::Ptr cloud_src(new PointCloud);
    //pcl::copyPointCloud(*keypointCloud, *cloud_src);

    //visualize_pcd(input_cloud, cloud_src);
    return (0);
}