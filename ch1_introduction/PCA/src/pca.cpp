#include "pca.h"
#include <iostream>

cv::Mat pca(cv::Mat img, int dim) {
  cv::resize(img, img, cv::Size(50, 50));
  cv::imshow("img", img);
  cv::waitKey(0);

  assert(img.channels() == 1);
  img.convertTo(img, CV_64FC1);

  cv::Mat img_vector = img.reshape(1, img.rows * img.cols);

  double img_aver = cv::sum(img)[0] / img_vector.rows;
  cv::Mat zero_mean_img_vector = img_vector - img_aver;

  cv::Mat cov_matrix(img_vector.rows, img_vector.rows, CV_64FC1);
  for (int i = 0; i < img_vector.rows; i++) {
    cov_matrix.at<double>(i, i) =
        img_vector.at<double>(i, 0) * img_vector.at<double>(i, 0);
  }

  cv::Mat e_values, e_vectors;
  cv::eigen(cov_matrix, e_values, e_vectors);

  cv::Mat main_component(
      cv::Mat::zeros(cov_matrix.rows, cov_matrix.rows, cov_matrix.type()));
  for (int i = 0; i < dim; i++) {
    main_component.at<double>(i, i) = cov_matrix.at<double>(i, i);
  }

  cv::Mat result;
  result = e_vectors * main_component;

  result = result.reshape(1, 50) + img_aver;
  return result;
}