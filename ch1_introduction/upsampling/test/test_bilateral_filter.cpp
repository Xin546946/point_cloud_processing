#include "bilateral_filter.h"
#include "opencv_utils.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char **argv) {
  cv::Mat rgb_img = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat depth_img = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

  cv::Mat vis;
  cv::vconcat(rgb_img, depth_img, vis);

  cv::imshow("top: rgb image; down: sparse depth image", vis);
  cv::waitKey(0);
  return 0;
}