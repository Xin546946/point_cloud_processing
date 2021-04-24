#include <pcl/registration/ia_ransac.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <time.h>
#include <pcl/common/io.h>
#include <iostream>
#include <pcl/keypoints/iss_3d.h>//关键点检测
#include <iss.h>

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
    //加载点云文件
    PointCloud::Ptr input_cloud(new PointCloud);//原点云，待配准
    if(pcl::io::loadPCDFile("/home/gfeng/gfeng_ws/point_cloud_processing/ch7_Feature_detection/data/0.pcd", *input_cloud) == -1){
        PCL_ERROR ("Couldn't read file\n");
        return (-1);
    }
    std::cout << "/////////////////////////////////////////////////" <<std::endl;
    std::cout << "原始点云数量："<<input_cloud->size() <<std::endl;

    clock_t start = clock();

    //pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoint(new pcl::PointCloud<pcl::PointXYZ>());
    ISSKeypoint iss;
    PointCloud::Ptr keypointscloud(new PointCloud);
    double model_solution = 0.4;//参数小，采取的关键点多，论文中为500左右

    //参数设置
    iss.useWeightedCovMat(true);
    iss.setLocalRadius(2.4);
    iss.setNonMaxRadius(1.6);
    iss.setThreshold(0.975, 0.975);
    iss.setMinNeighbors(5);
    iss.setInputPointCloud(input_cloud);
    iss.compute(keypointscloud);


    clock_t end = clock();
    cout << "iss关键点提取时间:" << (double)(end - start) / CLOCKS_PER_SEC << endl;
    cout << "iss关键点数量" << keypointscloud->size() << endl;
    
    PointCloud::Ptr cloud_src(new PointCloud);
    pcl::copyPointCloud(*keypointscloud, *cloud_src);

    //可视化
    visualize_pcd(input_cloud, cloud_src);
    return 0;
}
