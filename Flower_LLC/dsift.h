// opencv api
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

extern "C" 
{
#include <vl/generic.h>
#include <time.h>
#include <stdlib.h>
#include "vl/dsift.h"
#include "vl/pgm.h"
#include "vl/mathop.h"
#include "vl/imopv.h"
}

using namespace std;
using namespace cv;

Mat dsift(Mat img, int step, int binSize);
void meshgrid(const cv::Range &xgv, const cv::Range &ygv, int step, cv::Mat &X, cv::Mat &Y) ;
Mat normImg(Mat image, int maxImgSize);
Mat calculateSiftXY(Mat dfea, int width, int height, int patchSize, int step, bool flag);
void extractDsiftFeature(string databaseDir, string dsiftFeatureDir, int step, int binSize, int patchSize);