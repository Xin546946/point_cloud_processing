#include "bilateral_filter.h"
#include "display.h"
#include "opencv_utils.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char **argv) {
  // cv::Mat rgb_img = cv::imread(argv[1], cv::IMREAD_COLOR);
  // cv::Mat depth_img = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

  // cv::Mat jepmap_depth_img = apply_jetmap(depth_img);

  // cv::Mat vis;
  // cv::vconcat(rgb_img, jepmap_depth_img, vis);

  // cv::imshow("top: rgb image; down: sparse depth image", vis);
  // cv::waitKey(0);

  cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  cv::Mat cartoon_img = apply_bilateral_filter(img, 5, 5, 20.0, 20.0);

  // cv::Mat vis;
  cv::Mat vis = get_float_mat_vis_img(cartoon_img);
  // cv::vconcat(img, vis_cartoon, vis);

  cv::imshow("bilateral filter image", vis);
  cv::waitKey(0);

  return 0;
}