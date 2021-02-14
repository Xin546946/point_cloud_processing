#include "opencv_utils.h"
#include "pca.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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