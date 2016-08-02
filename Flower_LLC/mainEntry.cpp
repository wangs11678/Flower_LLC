#include "extract_sift.h"
#include "build_codebook.h"
#include "utils.h"
int main(int argc, char* argv[])
{
	/******************��ȡsift����***************************************************/
	cout<<"******************Extracting SIFT Feature*******************"<<endl;
	string databaseDir = "data\\train";
	vector<string> categories;
	GetDirList(databaseDir, &categories);
	const SiftDescriptorExtractor siftDetector;
	string siftImageDescriptorsDir = "result\\sift";

	ExtractSIFTFeature(databaseDir, categories, siftDetector, siftImageDescriptorsDir);
	
	/******************�����ֵ�*******************************************************/
	cout<<"******************Building Codebook***********************"<<endl;
	Mat vocabulary;
	string vocabularyFile = "result\\vocabulary.xml.gz";
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");

	vocabulary = BuildVocabulary(databaseDir, categories, detector, extractor, 1024);

	/******************LLC*******************************************************/

	return 0;
}



