#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

inline double compute_gaussian_pdf(double mean, double var, double sample) {
  return (1 / (sqrt(2 * M_PI) * var)) *
         exp(-0.5 * pow((sample - mean) / var, 2));
}

cv::Mat apply_bilateral_filter(cv::Mat img, int win_width = 3,
                               int win_height = 3, double sigma_position = 1.0,
                               double sigma_pixel = 3.0);