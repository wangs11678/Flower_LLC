#include "build_codebook.h"
#include "utils.h"
Mat BuildVocabulary(const string& databaseDir, 
					const vector<string>& categories, 
					const Ptr<FeatureDetector>& detector, 
					const Ptr<DescriptorExtractor>& extractor,
					int wordCount)
{
	Mat vocabulary;
	string vocabularyFile = "result\\vocabulary.xml.gz";
	FileStorage fs(vocabularyFile, FileStorage::READ);
	if(fs.isOpened())
	{
		fs["vocabulary"] >> vocabulary;
	}
	else
	{
		Mat allDescriptors;
		for (int index = 0; index != categories.size(); ++index)
		{
			cout << "processing category " << categories[index] << endl;
			string currentCategory = databaseDir + '\\' + categories[index];
			vector<string> filelist;
			GetFileList(currentCategory, &filelist);
			for (auto fileindex = filelist.begin(); fileindex != filelist.end(); fileindex++)
			{			
				string filepath = currentCategory + '\\' + *fileindex;
				Mat image = imread(filepath);
				if (image.empty())
				{
					continue; // maybe not an image file
				}
				vector<KeyPoint> keyPoints;
				Mat descriptors;
				detector -> detect(image, keyPoints);
				extractor -> compute(image, keyPoints, descriptors);
				if (allDescriptors.empty())
				{
					allDescriptors.create(0, descriptors.cols, descriptors.type());
				}
				allDescriptors.push_back(descriptors);
			}
			cout << "done processing category " << categories[index] << endl;
		}
		assert(!allDescriptors.empty());
		cout << "build vocabulary..." << endl;
		BOWKMeansTrainer bowTrainer(wordCount);
		vocabulary = bowTrainer.cluster(allDescriptors);
		//vocabulary = vocabulary.t();//ת��
		fs.open(vocabularyFile, FileStorage::WRITE);
		if (fs.isOpened())
		{
			//��vocabulary���󱣴���fs����ָ����yml�ļ���vocabulary��ǩ��
			fs << "vocabulary" << vocabulary;
		}
		cout << "done build vocabulary..." << endl;
	}
	return vocabulary;
}