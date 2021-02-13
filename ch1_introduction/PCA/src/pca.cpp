#include "pca.h"
#include <iostream>

cv::Mat pca(cv::Mat img, int dim) {
  cv::resize(img, img, cv::Size(50, 50));
  cv::imshow("img", img);
  cv::waitKey(0);

  assert(img.channels() == 1);
  img.convertTo(img, CV_64FC1);

  double img_aver = cv::sum(img)[0] / (img.rows * img.cols);
  cv::Mat zero_mean_img = img - img_aver;
  cv::Mat img_covar, img_mean;
  cv::calcCovarMatrix(zero_mean_img, img_covar, img_mean, CV_COVAR_ROWS);
  cv::Mat e_values, e_vectors;
  cv::eigen(img_covar, e_values, e_vectors);

  cv::Mat e_vectors_main_components = e_vectors.colRange(cv::Range(0, dim));

  cv::Mat result;
  result = zero_mean_img * e_vectors_main_components *
               e_vectors_main_components.t() +
           img_aver;

  return result;
}