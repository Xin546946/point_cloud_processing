#include "bilateral_filter.h"
#include "opencv_utils.h"
#include <iostream>

// todo computational cost need to be taken into consideration
cv::Mat apply_bilateral_filter(cv::Mat img, int size, double sigma_position,
                               double sigma_pixel) {
  assert(size % 2 == 1);
  if (img.type() != CV_64FC1) {
    img.convertTo(img, CV_64FC1);
  }

  cv::Mat result(cv::Mat::zeros(img.size(), img.type()));

  int win_half_size = (size - 1) / 2;

  for (int r = 0; r < img.rows; r++) {
    for (int c = 0; c < img.cols; c++) {
      double weight = 0.0;
      double local_weight = 0.0;
      for (int r_win = -win_half_size; r_win < win_half_size + 1; r_win++) {
        // r_win = std::min(std::max(0, r_win), img.rows - 1);

        for (int c_win = -win_half_size; c_win < win_half_size + 1; c_win++) {
          // c_win = std::min(std::max(0, c_win), img.cols - 1);
          double square_dist = std::pow(r_win, 2) + std::pow(c_win, 2);
          // r_win = std::min(std::max(0, r + r_win), img.rows - 1);
          // c_win = std::min(std::max(0, c + c_win), img.cols - 1);
          // if (r + r_win < 0 || c + c_win < 0) {
          //   continue;
          // }
          local_weight = compute_gaussian_pdf(0, sigma_pixel,
                                              img.at<double>(r_win, c_win) -
                                                  img.at<double>(r, c)) *
                         compute_gaussian_pdf(0, sigma_position, square_dist);
          result.at<double>(r, c) +=
              (local_weight * img.at<double>(r_win, c_win));
          weight += local_weight;
        }
        // std::cout << '\n';
      }
      result.at<double>(r, c) /= weight;
    }
    // std::cout << "finished 1 round" << '\n';
  }
  // cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
  return result;
}