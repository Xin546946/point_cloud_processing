#include "bilateral_filter.h"
#include "display.h"
#include "opencv_utils.h"
#include "tictoc.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
int main(int argc, char **argv) {
  cv::Mat img = read_img(argv[1], cv::IMREAD_GRAYSCALE);

  img.convertTo(img, CV_64FC1);
  cv::Mat sparse_depth = read_img(argv[2], cv::IMREAD_GRAYSCALE);
  sparse_depth.convertTo(sparse_depth, CV_64FC1);
  //   sparse_depth.convertTo(sparse_depth, CV_64FC1);

  cv::Mat upsapl_img =
      apply_bilateral_filter_for_upsampling(sparse_depth, 11, 7.0, 5.0);

  //   cv::Mat vis_upsamp = apply_jetmap(upsapl_img);
  //   cv::Mat vis_depth = apply_jetmap(sparse_depth);
  img.convertTo(img, CV_8UC1);
  cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
  assert(img.type() == CV_8UC3);
  cv::Mat jet_map_sparse_depth = apply_jetmap(sparse_depth);

  std::cout << img.type() << " " << jet_map_sparse_depth.type() << " " << '\n';
  cv::Mat vis_temp;
  cv::vconcat(read_img(argv[1], cv::IMREAD_COLOR), jet_map_sparse_depth,
              vis_temp);
  cv::Mat jet_map_depth = apply_jetmap(upsapl_img);
  //<< jet_map_depth.type() << '\n';
  cv::Mat vis;
  cv::vconcat(vis_temp, jet_map_depth, vis);
  cv::imshow("top: original image; middle: sparse depth map from pcl; bottom: "
             "upsampling",
             vis);
  cv::waitKey(0);
  return 0;
}