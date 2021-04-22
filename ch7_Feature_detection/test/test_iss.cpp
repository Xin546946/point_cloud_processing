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
    PointCloud::Ptr cloud_src_o(new PointCloud);//原点云，待配准
    if(pcl::io::loadPCDFile("/home/gfeng/gfeng_ws/point_cloud_processing/ch7_Feature_detection/data/0.pcd", *cloud_src_o) == -1){
        PCL_ERROR ("Couldn't read file\n");
        return (-1);
    }
    std::cout << "/////////////////////////////////////////////////" <<std::endl;
    std::cout << "原始点云数量："<<cloud_src_o->size() <<std::endl;
    //PointCloud::Ptr cloud_tgt_o(new PointCloud);//目标点云
    //pcl::io::loadPCDFile("E:/PointCloud/data/pc_4.pcd", *cloud_tgt_o);

    clock_t start = clock();
    //去除NAN点
    //std::vector<int> indices_src; //保存去除的点的索引
    //pcl::removeNaNFromPointCloud(*cloud_src_o, *cloud_src_o, indices_src);
    //std::cout << "remove *cloud_src_o nan" << cloud_src_o->size()<<endl;

    //std::vector<int> indices_tgt;
    //pcl::removeNaNFromPointCloud(*cloud_tgt_o, *cloud_tgt_o, indices_tgt);
    //std::cout << "remove *cloud_tgt_o nan" << cloud_tgt_o->size()<<endl;



    //iss关键点提取
    PointCloud::Ptr cloud_src_is(new PointCloud);
    //pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoint(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_det;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_1(new pcl::search::KdTree<pcl::PointXYZ>());

    double model_solution = 0.4;//参数小，采取的关键点多，论文中为500左右

    //参数设置
    iss_det.setSearchMethod(tree_1);
    iss_det.setSalientRadius(2.4);//
    iss_det.setNonMaxRadius(1.6);//
    iss_det.setThreshold21(0.975);
    iss_det.setThreshold32(0.975);
    iss_det.setMinNeighbors(5);
    iss_det.setNumberOfThreads(4);
    iss_det.setInputCloud(cloud_src_o);
    iss_det.compute(*cloud_src_is);


    clock_t end = clock();
    cout << "iss关键点提取时间:" << (double)(end - start) / CLOCKS_PER_SEC << endl;
    cout << "iss关键点数量" << cloud_src_is->size() << endl;
    
    PointCloud::Ptr cloud_src(new PointCloud);
    pcl::copyPointCloud(*cloud_src_is, *cloud_src);


    //可视化
    visualize_pcd(cloud_src_o, cloud_src);
    return 0;
}
