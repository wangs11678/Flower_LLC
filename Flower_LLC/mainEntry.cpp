#include "extract_sift.h"
#include "build_codebook.h"
#include "utils.h"
int main(int argc, char* argv[])
{
	/******************提取sift特征***************************************************/
	cout<<"******************Extracting SIFT Feature*******************"<<endl;
	string databaseDir = "data\\train";
	vector<string> categories;
	GetDirList(databaseDir, &categories);
	const SiftDescriptorExtractor siftDetector;
	string siftImageDescriptorsDir = "result\\sift";

	ExtractSIFTFeature(databaseDir, categories, siftDetector, siftImageDescriptorsDir);
	
	/******************构建字典*******************************************************/
	cout<<"******************Building Codebook***********************"<<endl;
	Mat vocabulary;
	string vocabularyFile = "result\\vocabulary.xml.gz";
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");

	vocabulary = BuildVocabulary(databaseDir, categories, detector, extractor, 1024);

	/******************LLC*******************************************************/

	return 0;
}



