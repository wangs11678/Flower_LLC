// opencv api
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;
void ExtractSIFTFeature(const string& databaseDir,
				 const vector<string>& categories, 
				 const SiftDescriptorExtractor detector,
				 const string& imageDescriptorsDir);