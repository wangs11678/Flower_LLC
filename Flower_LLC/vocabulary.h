#include "utils.h"
// opencv api
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
//#include <mat.h>

using namespace std;
using namespace cv;

Mat buildDictionary(const string& dsiftFeatureDir, const string& dictionaryFile, int wordCount);
//Mat readDictionary(string dictionaryFile);