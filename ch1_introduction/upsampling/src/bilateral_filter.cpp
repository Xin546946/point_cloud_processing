#include "bilateral_filter.h"
#include <iostream>

cv::Mat apply_bilateral_filter(cv::Mat img, int win_width, int win_height,
                               double sigma_position, double sigma_pixel) {
  assert(win_width % 2 == 1 && win_height % 2 == 1);
  if (img.type() != CV_64FC1) {
    img.convertTo(img, CV_64FC1);
  }

  cv::Mat result(cv::Mat::zeros(img.size(), img.type()));

  int win_half_width = (win_width - 1) / 2;
  int win_half_height = (win_height - 1) / 2;

  for (int r = 0; r < img.rows; r++) {
    for (int c = 0; c < img.cols; c++) {
      for (int r_win = r - win_half_height; r_win < r + win_half_height;
           r_win++) {
        for (int c_win = c - win_half_width; c_win < c + win_half_width;
             c_win++) {
          r_win = std::min(std::max(0, r_win), img.rows - 1);
          c_win = std::min(std::max(0, c_win), img.cols - 1);
          double square_dist = std::pow(r_win - r, 2) + std::pow(c_win - c, 2);
          result.at<double>(r, c) +=
              compute_gaussian_pdf(img.at<double>(r, c), sigma_pixel,
                                   img.at<double>(r_win, c_win)) *
              compute_gaussian_pdf(0, sigma_position, square_dist) *
              img.at<double>(r_win, c_win);
        }
      }
    }
  }
  cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
  return result;
}