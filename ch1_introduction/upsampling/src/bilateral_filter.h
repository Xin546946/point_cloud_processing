#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

inline double compute_gaussian_pdf(double mean, double var, double sample) {
  return (1 / (sqrt(2 * M_PI) * var)) *
         exp(-0.5 * pow((sample - mean) / var, 2));
}

cv::Mat apply_bilateral_filter(cv::Mat img, int size, double sigma_position,
                               double sigma_pixel);

cv::Mat apply_bilateral_filter_for_upsampling(cv::Mat img, int size,
                                              double sigma_position,
                                              double sigma_pixel);