#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream> 

using namespace std;
using namespace cv;

cv::Mat findKNN(cv::Mat &codebook, cv::Mat &input, int knn);
cv::Mat llccoding(cv::Mat &codebook, cv::Mat &input, int knn);

cv::Mat llcpooling(cv::Mat &tcodebook,
				   cv::Mat &tinput,
				   cv::Mat feaSet_x, 
				   cv::Mat feaSet_y, 
				   int width,
				   int height,
				   cv::Mat tllccodes);

void llc_coding_pooling(string databaseDir, 
						string dsiftFeatureDir, 
						string llcFeatureDir, 
						string feaTxt,
						cv::Mat &codebook, 
						int knn);