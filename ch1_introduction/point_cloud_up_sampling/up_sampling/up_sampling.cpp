#include "bilateral_filter.h"
#include <iostream>
#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>

int main(int argc, char **argv) {
  cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  //cv::resize(img, img, cv::Size(1000,200), 0, 0);

  img.convertTo(img, CV_32F);
  cv::Mat result = img.clone();

  //cv::bilateralFilter(img, result, 50, 20, 20); //TODO::learn this function code
  apply_filter(img, result, 7, 5.0, 5.0);

  //result.convertTo(result, CV_8UC1);
  //std::cout<<result;

  cv::normalize(result, result, 0, 1 ,cv::NORM_MINMAX);
 
  cv::imshow("filtered image", result);
  result = 255 * result;
  result.convertTo(result, CV_8UC1);
  //applyColorMap(result, result, cv::COLORMAP_JET);
  cv::imwrite("/home/gfeng/gfeng_ws/point_cloud_processing/ch1_introduction/point_cloud_up_sampling/data/depth_new.png", result);
  cv::waitKey(0);

  return 0;
}