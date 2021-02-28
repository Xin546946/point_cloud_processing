#include "bilateral_filter.h"
#include "opencv_utils.h"
#include <array>
#include <iostream>

// todo computational cost need to be taken into consideration
cv::Mat apply_bilateral_filter(cv::Mat img_, int size, double sigma_position,
                               double sigma_pixel) {

  assert(size % 2 == 1);
  cv::Mat img = img_.clone();
  if (img.type() != CV_8UC1) {
    img.convertTo(img, CV_8UC1);
  }

  cv::Mat dist_map = get_gaussian_kernel(size, sigma_position);
  cv::Mat img_64f;
  img.convertTo(img_64f, CV_64FC1);
  cv::Mat result(cv::Mat::zeros(img.size(), CV_64FC1));

  int win_half_size = (size - 1) / 2;
  double local_weight = 0.0;

  double pixel_coeff = -0.5 / (sigma_pixel * sigma_pixel);

  double gaussian_pixel_lut[256];
  for (int pixel_val = 0; pixel_val < 256; pixel_val++) {
    double pixel_val_double = static_cast<double>(pixel_val);
    gaussian_pixel_lut[pixel_val] =
        std::exp(pixel_val_double * pixel_val_double * pixel_coeff);
  }
  for (int r = 0; r < img.rows; r++) {
    for (int c = 0; c < img.cols; c++) {

      double weight = 0.0;

      for (int r_win = -win_half_size; r_win < win_half_size + 1; r_win++) {
        for (int c_win = -win_half_size; c_win < win_half_size + 1; c_win++) {

          if (r + r_win < 0 || c + c_win < 0 || r + r_win > img.rows - 1 ||
              c + c_win > img.cols - 1) {
            continue;
          }
          int r_local = r + r_win;
          int c_local = c + c_win;

          if (img.at<uchar>(r_local, c_local) >= img.at<uchar>(r, c)) {
            local_weight =
                gaussian_pixel_lut[std::abs(img.at<uchar>(r_local, c_local) -
                                            img.at<uchar>(r, c))] *
                dist_map.at<double>(r_win + win_half_size,
                                    c_win + win_half_size);
          } else {
            local_weight =
                gaussian_pixel_lut[std::abs(img.at<uchar>(r, c) -
                                            img.at<uchar>(r_local, c_local))] *
                dist_map.at<double>(r_win + win_half_size,
                                    c_win + win_half_size);
          }

          //  compute_gaussian_pdf(0, sigma_position, square_dist);
          result.at<double>(r, c) +=
              (local_weight * img_64f.at<double>(r_local, c_local));
          weight += local_weight;
        }
        // std::cout << '\n';
      }
      result.at<double>(r, c) /= weight;
    }
    // std::cout << "finished 1 round" << '\n';
  }
  // result.convertTo(result, CV_64FC1);
  return result;
}

cv::Mat apply_bilateral_filter_for_upsampling(cv::Mat img_, int size,
                                              double sigma_position,
                                              double sigma_pixel) {
  cv::Mat img = img_.clone();

  assert(size % 2 == 1);

  if (img.type() != CV_8UC1) {
    img.convertTo(img, CV_8UC1);
  }

  cv::Mat dist_map = get_gaussian_kernel(size, sigma_position);
  cv::Mat img_64f;
  img.convertTo(img_64f, CV_64FC1);
  cv::Mat result(cv::Mat::zeros(img.size(), CV_64FC1));

  int win_half_size = (size - 1) / 2;
  double local_weight = 0.0;

  double pixel_coeff = -0.5 / (sigma_pixel * sigma_pixel);

  double gaussian_pixel_lut[256];
  for (int pixel_val = 0; pixel_val < 256; pixel_val++) {
    double pixel_val_double = static_cast<double>(pixel_val);
    gaussian_pixel_lut[pixel_val] =
        std::exp(pixel_val_double * pixel_val_double * pixel_coeff);
  }

  for (int r = 0; r < img.rows; r++) {
    for (int c = 0; c < img.cols; c++) {
      if (img.at<uchar>(r, c) != 0) {
        continue;
      }
      double weight = 0.0;

      for (int r_win = -win_half_size; r_win < win_half_size + 1; r_win++) {
        for (int c_win = -win_half_size; c_win < win_half_size + 1; c_win++) {

          if (r + r_win < 0 || c + c_win < 0 || r + r_win > img.rows - 1 ||
              c + c_win > img.cols - 1) {
            continue;
          }

          if (img.at<uchar>(r + r_win, c + c_win) == 0) {
            continue;
          }
          int r_local = r + r_win;
          int c_local = c + c_win;

          if (img.at<uchar>(r_local, c_local) >= img.at<uchar>(r, c)) {
            local_weight =
                gaussian_pixel_lut[std::abs(img.at<uchar>(r_local, c_local) -
                                            img.at<uchar>(r, c))] *
                dist_map.at<double>(r_win + win_half_size,
                                    c_win + win_half_size);
          } else {
            local_weight =
                gaussian_pixel_lut[std::abs(img.at<uchar>(r, c) -
                                            img.at<uchar>(r_local, c_local))] *
                dist_map.at<double>(r_win + win_half_size,
                                    c_win + win_half_size);
          }

          //  compute_gaussian_pdf(0, sigma_position, square_dist);
          result.at<double>(r, c) +=
              (local_weight * img_64f.at<double>(r_local, c_local));
          weight += local_weight;
        }
        // std::cout << '\n';
      }
      result.at<double>(r, c) /= weight;
    }
  }

  return result;
}