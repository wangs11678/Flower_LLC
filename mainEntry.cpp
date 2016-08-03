#include "extract_sift.h"
#include "build_codebook.h"
#include "llc_code.h"
#include "utils.h"

int main(int argc, char* argv[])
{
	/****************** Extracting SIFT Feature *************************************************/
	cout<<"****************** Extracting SIFT Feature *******************"<<endl;
	string databaseDir = "data\\train";
	vector<string> categories;
	GetDirList(databaseDir, &categories);
	const SiftDescriptorExtractor siftDetector;
	string siftImageDescriptorsDir = "result/sift";
	MakeDir(siftImageDescriptorsDir);

	ExtractSIFTFeature(databaseDir, categories, siftDetector, siftImageDescriptorsDir);
	cout<<"****************** Extracting SIFT Feature end ***************"<<endl;

	/****************** Building Codebook *******************************************************/
	cout<<"****************** Building Codebook *************************"<<endl;
	Mat vocabulary;
	string vocabularyFile = "result\\vocabulary.xml.gz";
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");

	vocabulary = BuildVocabulary(databaseDir, categories, detector, extractor, 1024);
	cout<<"****************** Building Codebook end *********************"<<endl;

	/****************** Locality-constrained Linear Coding ***************************************/
	cout<<"****************** Locality-constrained Linear Coding ********"<<endl;
	int k = 5;
	llc(vocabulary, 5);
	cout<<"****************** Locality-constrained Linear Coding end ****"<<endl;

	return 0;
}



