#include "pca.h"

cv::Mat pca(cv::Mat img, int dim) {
  assert(img.channels() == 1);
  img.convertTo(img, CV_64FC1);

  cv::Mat img_vector = img.reshape(img.rows * img.cols, 1);

  cv::Mat zero_mean_img_vector =
      img_vector - img_vector / cv::sum(img_vector)[0];
  cv::Mat cov_matrix = zero_mean_img_vector.mul(zero_mean_img_vector.t());
  cv::imshow("cov matrix", cov_matrix);
  cv::waitKey(0);

  cv::Mat e_values, e_vectors;
  cv::eigen(cov_matrix, e_values, e_vectors);

  cv::Mat vis_e_values;
  cv::normalize(e_values, vis_e_values, cv::NORM_MINMAX);
  cv::imshow("e_values", vis_e_values);
  cv::waitKey(0);
  cv::Mat vis_e_vectors;
  cv::normalize(e_vectors, vis_e_vectors, cv::NORM_MINMAX);
  cv::imshow("e_vectors", vis_e_vectors);
  cv::waitKey(0);

  cv::Mat main_component(cv::Mat::zeros(cov_matrix.size(), cov_matrix.type()));
  for (int i = 0; i < dim; i++) {
    main_component.at<double>(i, i) = cov_matrix.at<double>(i, i);
  }

  cv::Mat result;
  result = e_vectors.mul(main_component.mul(e_vectors.t()));

  return result;
}