#include "bilateral_filter.h"
#include "display.h"
#include "opencv_utils.h"
#include "tictoc.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char **argv) {
  std::vector<cv::Mat> depths;

  for (int id = 1; id < 101; id++) {
    cv::Mat depth =
        read_img(argv[1] + std::to_string(id) + ".png", cv::IMREAD_GRAYSCALE);
    depths.push_back(depth);
  }
  std::cout << depths.size() << '\n';
  cv::Mat upsampl_img;

  std::vector<cv::Mat> upsampl_imgs;

  for (int id = 1; id < 101; id++) {
    tictoc::tic();
    upsampl_img =
        apply_bilateral_filter_for_upsampling(depths[id - 1], 11, 7.0, 5.0);
    std::cout << std::to_string(id) << "-th frame costs " << tictoc::toc() / 1e6
              << "s" << '\n';
    cv::Mat jet_map_depth = apply_jetmap(upsampl_img);
    cv::Mat ground_truth =
        read_img(argv[2] + std::to_string(id) + ".png", cv::IMREAD_GRAYSCALE);
    cv::Mat jet_map_ground_truth = apply_jetmap(ground_truth);
    cv::Mat vis;
    cv::vconcat(jet_map_depth, jet_map_ground_truth, vis);
    cv::imshow("up: my upsampling; down: groundtruth", vis);
    cv::waitKey(1);

    cv::imwrite(
        "../datas/depth_selection/val_selection_cropped/prediction_depth/" +
            std::to_string(id) + ".png",
        upsampl_img);
    // std::cout << "Finish processing the " << id << "-th depth image." <<
    // '\n';
  }

  return 0;
}