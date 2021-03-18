#include "math_utils.h"
#include "opencv_utils.h"
#include "pca.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

std::vector<cv::Point2f> generate_ring_dataset(int num_points, int dick,
                                               int r_center, int c_center,
                                               float radius) {
  assert(dick % 2 != 0);
  double points_resolution = num_points / 10;
  double angle = 2 * M_PI * 10 / num_points;
  std::vector<cv::Point2f> datas(num_points);
  int half_dick = (dick + 1) / 2;
  // cv::Mat datas(cv::Mat::zeros(cv::Size(2, num_points), CV_64FC1));
  for (int i = 0; i < points_resolution; i++) {
    for (int sub_i = -half_dick; sub_i < dick; sub_i++) {
      datas.emplace_back(c_center + (radius + sub_i) * cos(i * angle),
                         r_center + (radius + sub_i) * sin(i * angle));
    }
  }
}

int main(int argc, char **argv) {
  cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

  int pca_dim = 40;
  cv::Mat pca_output;

  if (pca_dim <= std::min(img.rows, img.cols)) {
    pca_output = pca(img, pca_dim);
  }
  cv::Mat vis_pca_output;
  cv::normalize(pca_output, vis_pca_output, 0, 1, cv::NORM_MINMAX);
  cv::imshow("pca image", vis_pca_output);
  cv::waitKey(0);

  return 0;
}