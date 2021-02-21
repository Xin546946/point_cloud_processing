#include "bilateral_filter.h"
#include "display.h"
#include "opencv_utils.h"
#include "tictoc.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// todo bilateral filter for upsampling: for depth(r,c) if the value is zero
// ((r,c) need to do upsamp), the value near 0 in the neighborhood has big
// weight???
int main(int argc, char **argv) {
  cv::Mat img = read_img(argv[1], cv::IMREAD_GRAYSCALE);
  img.convertTo(img, CV_64FC1);
  cv::Mat sparse_depth = read_img(argv[2], cv::IMREAD_GRAYSCALE);
  sparse_depth.convertTo(sparse_depth, CV_64FC1);
  //   sparse_depth.convertTo(sparse_depth, CV_64FC1);

  cv::Mat upsapl_img =
      apply_bilateral_filter_for_upsampling(sparse_depth, 51, 31.0, 31.0);
  //   cv::Mat vis_upsamp = apply_jetmap(upsapl_img);
  //   cv::Mat vis_depth = apply_jetmap(sparse_depth);
  cv::Mat vis_tmp;
  cv::Mat vis;
  cv::Mat img_color;
  img.convertTo(img_color, CV_8UC3);
  std::cout << img.type() << " " << sparse_depth.type() << " "
            << upsapl_img.type() << '\n';
  cv::Mat vis_img = get_float_mat_vis_img(img);
  cv::vconcat(vis_img, sparse_depth, vis_tmp);
  cv::vconcat(vis_tmp, upsapl_img, vis);
  cv::imshow("upsampling of depth image", vis);
  cv::waitKey(0);
  return 0;
}