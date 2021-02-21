#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>

double computeGaussian(int sigma_squared, int x_squared);
void apply_filter(const cv::Mat& image, cv::Mat des, int halfFilterSize, int sigma1, int sigma2);
