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

  cv::Mat img = read_img(argv[1], cv::IMREAD_GRAYSCALE);
  img.convertTo(img, CV_64FC1);
  // cv::resize(img, img, cv::Size(100, 30));
  cv::Mat cartoon_img = apply_bilateral_filter(img, 5, 21.0, 21.0);

  cv::Mat vis;
  cv::Mat vis_cartoon = get_float_mat_vis_img(cartoon_img);
  cv::Mat vis_img = get_float_mat_vis_img(img);
  cv::vconcat(vis_img, vis_cartoon, vis);

  cv::imshow("bilateral filter image", vis);
  cv::waitKey(0);

  cv::Mat opencv_bilateral_image;

  img.convertTo(img, CV_32FC1);
  cv::bilateralFilter(img, opencv_bilateral_image, 5, 21.0, 21.0);
  cv::Mat vis_opencv = get_float_mat_vis_img(opencv_bilateral_image);
  vis_opencv.convertTo(vis_opencv, CV_32FC1);
  vis_img.convertTo(vis_img, CV_32FC1);
  cv::vconcat(vis_img, vis_opencv, vis);
  cv::imshow("bilateral filter image OPENCV", vis);
  cv::waitKey(0);
  return 0;
}