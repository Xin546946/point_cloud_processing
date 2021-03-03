#include "bilateral_filter.h"
#include <iostream>
#include <string>
#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>

int main(int argc, char **argv){
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> gts;
    for(int num = 1; num < 101; num++){
        std::string path = argv[1] + std::to_string(num) + ".png";
        std::string path2 = argv[2] + std::to_string(num) + ".png";
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        cv::Mat gt = cv::imread(path2, cv::IMREAD_GRAYSCALE);
        if(!img.data || !gt.data) {
            std::cout<<"Error while loading images!";
            return 0;
        }
        images.push_back(img);
        gts.push_back(gt);
    }
    std::cout<<"there are "<<images.size()<<" images in total"<<'\n';
    //cv::resize(img, img, cv::Size(1000,200), 0, 0);
    int cur = 1;
    for(cv::Mat img : images){
        img.convertTo(img, CV_32F);
        cv::Mat ans = img.clone();
        apply_filter(img, ans, 7, 5.0, 5.0);
        ans.convertTo(ans, CV_8UC1);
        std::string outPath = "/home/gfeng/gfeng_ws/point_cloud_processing/ch1_introduction/point_cloud_up_sampling/data/prediction/" + std::to_string(cur) + ".png";
        cv::imwrite(outPath, ans);

        cv::Mat jet_ans;
        cv::applyColorMap(ans, jet_ans, cv::COLORMAP_JET);

        cv::Mat groud = gts[cur-1];
        cv::Mat groud_jet;
        cv::applyColorMap(groud, groud_jet, cv::COLORMAP_JET);
        cv::Mat vis;
        cv::vconcat(jet_ans, groud_jet, vis);
        cv::imshow("jet image", vis);
        cv::waitKey(1);

        std::cout<<cur<<" images completed"<<'\n';
        cur++;
    }
    return 0;
}