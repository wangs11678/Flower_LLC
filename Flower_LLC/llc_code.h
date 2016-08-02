#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

cv::Mat findKNN(cv::Mat &codebook, cv::Mat &input, int k);
cv::Mat llccode(cv::Mat &codebook, cv::Mat &input, cv::Mat IDX, int k);