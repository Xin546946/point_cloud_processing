#include "bilateral_filter.h"
#include <iostream>



double computeGaussian(int sigma_squared, int x_squared){
    double res = (1 / sqrt(2 * M_PI * sigma_squared)) * exp(- x_squared / (2 * sigma_squared));
    return res;
}



void apply_filter(const cv::Mat& image, cv::Mat des, int halfFilterSize, int sigma1, int sigma2){//give float img
    if(image.type() != CV_32F){
        std::cout<<"please pass float images as input!";
        return;
    }
    int width = image.cols;
    int height = image.rows;
    for(int r = 0; r < height; r++){
        for(int c = 0; c < width; c++){
            if(image.at<float>(r, c) != 0.0) continue;
            //apply double gaussian
            double sum1 = 0;
            double sum2 = 0;
            for(int r_win = -halfFilterSize; r_win <= halfFilterSize; r_win++){
                for(int c_win = -halfFilterSize; c_win <= halfFilterSize; c_win++){
                    if(r + r_win < 0 || c + c_win < 0 || r + r_win >= height || c + c_win >= width || image.at<float>(r+r_win, c+c_win) == 0.0) continue;
                    int posDistSquared = pow(r_win, 2) + pow(c_win, 2);
                    int depthDistSquared = pow(image.at<float>(r,c) - image.at<float>(r + r_win, c + c_win), 2);
                    sum1 += computeGaussian(sigma1, posDistSquared) * computeGaussian(sigma2, depthDistSquared) * image.at<float>(r + r_win , c + c_win);
                    sum2 += computeGaussian(sigma1, posDistSquared) * computeGaussian(sigma2, depthDistSquared);
                }
            }
            des.at<float>(r,c) = sum1 / sum2;
        }
    }
}