// windows api
#include <Windows.h>
#include <tchar.h>
#include <strsafe.h>
#pragma comment( lib, "User32.lib")
// c api
#include <stdio.h>
#include <string.h>
#include <assert.h>
// c++ api
#include <string>
#include <map>
#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
// opencv api
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"


using namespace std;
using namespace cv;
// some utility functions
void MakeDir( const string& filepath );
void help( const char* progName );
void GetDirList( const string& directory, vector<string>* dirlist );
void GetFileList( const string& directory, vector<string>* filelist );

const string kVocabularyFile( "vocabulary.xml.gz" );
const string kBowImageDescriptorsDir( "/bagOfWords" );
const string kSvmsDirs( "/svms" );

class Params {
public:
	Params(): wordCount( 1000 ), detectorType( "SIFT" ),
			descriptorType( "SIFT" ), matcherType( "FlannBased" ){ }

	int		wordCount;
	string	detectorType;
	string	descriptorType;
	string	matcherType;
};

/*
 * loop through every directory 
 * compute each image's keypoints and descriptors
 * train a vocabulary
 */
Mat BuildVocabulary( const string& databaseDir, 
					 const vector<string>& categories, 
					 const Ptr<FeatureDetector>& detector, 
					 const Ptr<DescriptorExtractor>& extractor,
					 int wordCount) {
	Mat allDescriptors;
	for ( int index = 0; index != categories.size(); ++index ) {
		cout << "processing category " << categories[index] << endl;
		string currentCategory = databaseDir + '\\' + categories[index];
		vector<string> filelist;
		GetFileList( currentCategory, &filelist);
		for ( auto fileindex = filelist.begin(); fileindex != filelist.end(); ++fileindex ) {			
			string filepath = currentCategory + '\\' + *fileindex;
			Mat image = imread( filepath );
			if ( image.empty() ) {
				continue; // maybe not an image file
			}
			vector<KeyPoint> keyPoints;
			Mat descriptors;
			detector -> detect( image, keyPoints);
			extractor -> compute( image, keyPoints, descriptors );
			if ( allDescriptors.empty() ) {
				allDescriptors.create( 0, descriptors.cols, descriptors.type() );
			}
			allDescriptors.push_back( descriptors );
		}
		cout << "done processing category " << categories[index] << endl;
	}
	assert( !allDescriptors.empty() );
	cout << "build vocabulary..." << endl;
	BOWKMeansTrainer bowTrainer( wordCount );
	Mat vocabulary = bowTrainer.cluster( allDescriptors );
	cout << "done build vocabulary..." << endl;
	return vocabulary;
}

// bag of words of an image as its descriptor, not keypoint descriptors
void ComputeBowImageDescriptors( const string& databaseDir,
								 const vector<string>& categories, 
								 const Ptr<FeatureDetector>& detector,
								 Ptr<BOWImgDescriptorExtractor>& bowExtractor,
								 const string& imageDescriptorsDir,
								 map<string, Mat>* samples) {	
	for ( auto i = 0; i != categories.size(); ++i ) {
		string currentCategory = databaseDir + '\\' + categories[i];
		vector<string> filelist;
		GetFileList( currentCategory, &filelist);	
		for ( auto fileitr = filelist.begin(); fileitr != filelist.end(); ++fileitr ) {
			string descriptorFileName = imageDescriptorsDir + "\\" + ( *fileitr ) + ".xml.gz";
			FileStorage fs( descriptorFileName, FileStorage::READ );
			Mat imageDescriptor;
			if ( fs.isOpened() ) { // already cached
				fs["imageDescriptor"] >> imageDescriptor;
			} else {
				string filepath = currentCategory + '\\' + *fileitr;
				Mat image = imread( filepath );
				if ( image.empty() ) {
					continue; // maybe not an image file
				}
				vector<KeyPoint> keyPoints;
				detector -> detect( image, keyPoints );
				bowExtractor -> compute( image, keyPoints, imageDescriptor );
				fs.open( descriptorFileName, FileStorage::WRITE );
				if ( fs.isOpened() ) {
					fs << "imageDescriptor" << imageDescriptor;
				}
			}
			if ( samples -> count( categories[i] ) == 0 ) {
				( *samples )[categories[i]].create( 0, imageDescriptor.cols, imageDescriptor.type() );
			}
			( *samples )[categories[i]].push_back( imageDescriptor );
		}
	}
}

void TrainSvm( const map<string, Mat>& samples, const string& category, const CvSVMParams& svmParams, CvSVM* svm ) {
	Mat allSamples( 0, samples.at( category ).cols, samples.at( category ).type() );
	Mat responses( 0, 1, CV_32SC1 );
	//assert( responses.type() == CV_32SC1 );
	allSamples.push_back( samples.at( category ) );
	Mat posResponses( samples.at( category ).rows, 1, CV_32SC1, Scalar::all(1) ); 
	responses.push_back( posResponses );
	
	for ( auto itr = samples.begin(); itr != samples.end(); ++itr ) {
		if ( itr -> first == category ) {
			continue;
		}
		allSamples.push_back( itr -> second );
		Mat response( itr -> second.rows, 1, CV_32SC1, Scalar::all( -1 ) );
		responses.push_back( response );
		
	}
	svm -> train( allSamples, responses, Mat(), Mat(), svmParams );
}

// using 1-vs-all method, train an svm for each category.
// choose the category with the biggest confidence
string ClassifyBySvm( const Mat& queryDescriptor, const map<string, Mat>& samples, const string& svmDir ) {
	string category;
	SVMParams svmParams;
	int sign = 0; //sign of the positive class
	float confidence = -FLT_MAX;
	for ( auto itr = samples.begin(); itr != samples.end(); ++itr ) {
		CvSVM svm;
		string svmFileName = svmDir + "\\" + itr -> first + ".xml.gz";
		FileStorage fs( svmFileName, FileStorage::READ );
		if ( fs.isOpened() ) { // exist a previously trained svm
			fs.release();
			svm.load( svmFileName.c_str() );
		} else {
			TrainSvm( samples, itr->first, svmParams, &svm );
			if ( !svmDir.empty() ) {
				svm.save( svmFileName.c_str() );
			}
		}
		// determine the sign of the positive class
		if ( sign == 0 ) {
			float scoreValue = svm.predict( queryDescriptor, true );
			float classValue = svm.predict( queryDescriptor, false );
			sign = ( scoreValue < 0.0f ) == ( classValue < 0.0f )? 1 : -1;
		}
		float curConfidence = sign * svm.predict( queryDescriptor, true );
		if ( curConfidence > confidence ) {
			confidence = curConfidence;
			category = itr -> first;
		}
	}
	return category;
}

string ClassifyByMatch( const Mat& queryDescriptor, const map<string, Mat>& samples ) {
	// find the best match and return category of that match
	int normType = NORM_L2;
	Ptr<DescriptorMatcher> histogramMatcher = new BFMatcher( normType );
	float distance = FLT_MAX;
	struct Match{
		string category;
		float distance;
		Match(string c, float d): category( c ), distance( d ){}
		bool operator<( const Match& rhs ) const{ 
			return distance > rhs.distance; 
		}
	};
	priority_queue<Match, vector<Match> > matchesMinQueue;
	const int numNearestMatch = 9;
	for ( auto itr = samples.begin(); itr != samples.end(); ++itr ) {
		vector<vector<DMatch> > matches;
		histogramMatcher -> knnMatch( queryDescriptor, itr ->second, matches, numNearestMatch );
		for ( auto itr2 = matches[0].begin(); itr2 != matches[0].end(); ++ itr2 ) {
			matchesMinQueue.push( Match( itr -> first, itr2 -> distance ) );
		}
	}
	string category;
	int maxCount = 0;
	map<string, size_t> categoryCounts;
	size_t select = std::min( static_cast<size_t>( numNearestMatch ), matchesMinQueue.size() );
	for ( size_t i = 0; i < select; ++i ) {
		string& c = matchesMinQueue.top().category;
		++categoryCounts[c];
		int currentCount = categoryCounts[c];
		if ( currentCount > maxCount ) {
			maxCount = currentCount;
			category = c;
		}
		matchesMinQueue.pop();
	}
	return category;
}

int main( int argc, char* argv[] ) {

	if ( argc != 5 && argc != 8 ) {
		help( argv[0] );
		return -1;
	}
	// read params
	Params params;
	string method = argv[1];
	string queryImage = argv[2];
	string databaseDir = argv[3];
	string resultDir = argv[4];
	if ( argc == 8 ) {
		params.detectorType = argv[5];
		params.descriptorType = argv[6];
		params.matcherType = argv[7];
	}

	//string method = "svm";
	//string queryImage = "data\\test";
	//string databaseDir = "data\\train";
	//string resultDir = "result";

	//params.descriptorType = "SIFT";

	cv::initModule_nonfree();

	string bowImageDescriptorsDir = resultDir + kBowImageDescriptorsDir;
	string svmsDir = resultDir + kSvmsDirs;
	MakeDir( resultDir );
	MakeDir( bowImageDescriptorsDir );
	MakeDir( svmsDir );

	// key: image category name
	// value: histogram of image
	vector<string> categories;
	GetDirList( databaseDir, &categories );
	
	Ptr<FeatureDetector> detector = FeatureDetector::create( params.descriptorType );
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create( params.descriptorType );
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( params.matcherType );
	if ( detector.empty() || extractor.empty() || matcher.empty() ) {
		cout << "feature detector or descriptor extractor or descriptor matcher cannot be created.\nMaybe try other types?" << endl;
	}
	Mat vocabulary;
	string vocabularyFile = resultDir + '\\' + kVocabularyFile;
	FileStorage fs( vocabularyFile, FileStorage::READ );
	if ( fs.isOpened() ) {
		fs["vocabulary"] >> vocabulary;
	} else {
		vocabulary = BuildVocabulary( databaseDir, categories, detector, extractor, params.wordCount );
		FileStorage fs( vocabularyFile, FileStorage::WRITE );
		if ( fs.isOpened() ) {
			fs << "vocabulary" << vocabulary;
		}
	}
	Ptr<BOWImgDescriptorExtractor> bowExtractor = new BOWImgDescriptorExtractor( extractor, matcher );
	bowExtractor -> setVocabulary( vocabulary );
	map<string, Mat> samples;//key: category name, value: histogram
	
	ComputeBowImageDescriptors( databaseDir, categories, detector, bowExtractor, bowImageDescriptorsDir,  &samples );
	
	cout << "Classify image " << queryImage << "." << endl;
	Mat image = imread( queryImage );
	vector<KeyPoint> keyPoints;
	detector -> detect( image, keyPoints );
	Mat queryDescriptor;
	bowExtractor -> compute( image, keyPoints, queryDescriptor );
	string category;
	if ( method == "svm" ) {
		category = ClassifyBySvm( queryDescriptor, samples, svmsDir );
	} else {
		category = ClassifyByMatch( queryDescriptor, samples );
	}
	cout << "I think it should be " << category << "." << endl;
	getchar();
	return 0;
}

void help( const char* progName ) {
	cout << "Usage: \n"
		 << progName << " {classify method} {query image} {image set path} {result directory}\n"
		 << "or: \n"
		 << progName << " {classify method} {image set path} {result directory} {feature detector} {descriptor extrator} {descriptor matcher}\n"
		 << "\n"
		 << "Input parameters: \n"
		 << "{classify method}			\n	Method used to classify image, can be one of svm or match.\n"
		 << "{query image}				\n	Path to query image.\n"
		 << "{image set path}			\n	Path to image training set, organized into categories, like Caltech 101.\n"
		 << "{result directory}			\n	Path to result directory.\n"
		 << "{feature detector}			\n	Feature detector name, should be one of\n"
		 <<	"	FAST, STAR, SIFT, SURF, MSER, GFTT, HARRIS.\n"
		 << "{descriptor extractor}		\n	Descriptor extractor name, should be one of \n"
		 <<	"	SURF, OpponentSIFT, SIFT, OpponentSURF, BRIEF.\n"
		 << "{descriptor matcher}		\n	Descriptor matcher name, should be one of\n"
		 << "	BruteForce, BruteForce-L1, FlannBased, BruteForce-Hamming, BruteForce-HammingLUT.\n";
}

void MakeDir( const string& filepath ) {
	TCHAR path[MAX_PATH];
#ifdef _UNICODE
	MultiByteToWideChar(CP_ACP, NULL, filepath.c_str(), -1, path, MAX_PATH);
#else
	StringCchCopy( path, MAX_PATH, filepath.c_str() );
#endif
	CreateDirectory( path, 0 );
}

void ListDir( const string& directory, bool ( *filter )( const WIN32_FIND_DATA& entry ), vector<string>* entries) {
	WIN32_FIND_DATA entry;
	TCHAR dir[MAX_PATH];
	HANDLE hFind = INVALID_HANDLE_VALUE;
#ifdef _UNICODE
	MultiByteToWideChar(CP_ACP, NULL, directory.c_str(), -1, dir, MAX_PATH);
	char dirName[MAX_PATH];
#endif
	StringCchCat( dir, MAX_PATH, _T( "\\*" ) );
	hFind = FindFirstFile( dir, &entry );
	do {
		if ( filter( entry ) ) {
#ifdef _UNICODE
			WideCharToMultiByte( CP_ACP, NULL, entry.cFileName, -1, dirName, MAX_PATH, NULL, NULL );
			entries -> push_back( dirName );
#else
			entries -> push_back( entry.cFileName );
#endif
		}
	} while ( FindNextFile( hFind, &entry ) != 0 );
}

void GetDirList( const string& directory, vector<string>* dirlist ) {
	ListDir( directory, []( const WIN32_FIND_DATA& entry ){ 
							return ( entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY ) && 
								   lstrcmp( entry.cFileName, _T( "." ) ) != 0 &&
								   lstrcmp( entry.cFileName, _T( ".." ) ) != 0;}, dirlist);
}

void GetFileList( const string& directory, vector<string>* filelist ) {
	ListDir( directory, []( const WIN32_FIND_DATA& entry ){ 
							return !( entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY );}, filelist);
}