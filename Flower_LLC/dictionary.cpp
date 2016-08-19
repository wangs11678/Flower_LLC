#include "dictionary.h"

/*
 * 从提取的dsift特征中读取descriptors进行聚类
 * train a dictionary
 */
Mat buildDictionary(const string& dsiftFeatureDir, const string& dictionaryFile, int wordCount)
{
	Mat dictionary; 
	FileStorage fs(dictionaryFile, FileStorage::READ);
	if(fs.isOpened())
	{
		fs["dictionary"] >> dictionary;
	}
	else
	{
		vector<string> categories;
		GetDirList(dsiftFeatureDir, &categories);
		Mat allDescriptors;
		for (int index = 0; index != categories.size(); index++)
		{
			cout << "Processing category " << categories[index] << endl;
			string currentCategory = dsiftFeatureDir + '\\' + categories[index];
			vector<string> filelist;
			GetFileList(currentCategory, &filelist);
			for (auto fileindex = filelist.begin(); fileindex != filelist.end(); fileindex += 5)
			{			
				string filepath = currentCategory + '\\' + (*fileindex);

				Mat descriptors;
				FileStorage fs2(filepath, FileStorage::READ);
				if(fs2.isOpened())
				{
					fs2["dsiftFeature"] >> descriptors; //从提取的dsift特征里读取特征到descriptors
				}
				if (allDescriptors.empty())
				{
					allDescriptors.create(0, descriptors.cols, descriptors.type()); 
				}
				allDescriptors.push_back(descriptors); //将所有读取的dsift特征挨个存放到allDescriptors
			}
		}
		assert(!allDescriptors.empty());
		cout << "build dictionary..." << endl;
		BOWKMeansTrainer bowTrainer(wordCount);
		dictionary = bowTrainer.cluster(allDescriptors); //进行聚类

		fs.open(dictionaryFile, FileStorage::WRITE);
		if (fs.isOpened())
		{
			//将vocabulary矩阵保存在fs对象指定的xml文件的vocabulary标签下
			fs << "dictionary" << dictionary;
		}
		cout << "done build dictionary..." << endl;
	}	
	return dictionary;
}