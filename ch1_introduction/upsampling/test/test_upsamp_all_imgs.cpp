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

  cv::Mat upsampl_img;

  for (int id = 1; id < 101; id++) {

    upsampl_img =
        apply_bilateral_filter_for_upsampling(depths[id], 11, 7.0, 5.0);
    cv::imwrite(
        "../datas/depth_selection/val_selection_cropped/prediction_depth" +
            std::to_string(id) + ".png",
        upsampl_img);
    std::cout << "Finish processing the " << id << "-th depth image." << '\n';
  }

  return 0;
}