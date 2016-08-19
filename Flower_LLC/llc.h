#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

cv::Mat findKNN(cv::Mat &codebook, cv::Mat &input, int knn);
cv::Mat llccoding(cv::Mat &codebook, cv::Mat &input, int knn);
cv::Mat llcpooling(cv::Mat &codebook, cv::Mat &input, int knn, cv::Mat llccodes, string imageFileName);

void llc_coding_pooling(string databaseDir, string dsiftFeatureDir, string llcFeatureDir, cv::Mat &codebook, int knn);