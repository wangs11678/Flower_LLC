#include "vocabulary.h"

/*
Mat readDictionary(string dictionaryFile)
{
	Mat dictionary(1024, 128, CV_32FC1);; 
	FileStorage fs(dictionaryFile, FileStorage::READ);
	if(fs.isOpened())
	{
		fs["dictionary"] >> dictionary;
	}
	else
	{
		MATFile *pmatFile = NULL;  
		mxArray *pMxArray = NULL;  
  
		// ��ȡ.mat�ļ�������mat�ļ���Ϊ"Caltech101_SIFT_Kmeans_1024.mat"�����а���"B"��  
		double *B;  
  
		pmatFile = matOpen("result\\Caltech101_SIFT_Kmeans_1024.mat","r");  
		pMxArray = matGetVariable(pmatFile, "B");  
		B = (double*) mxGetData(pMxArray);  
		int M = mxGetM(pMxArray);  
		int N = mxGetN(pMxArray);  

		//Matrix<double> A(M,N);  
		for (int i = 0; i < M; i++)  
		{
			for (int j = 0; j < N; j++)  
			{
				dictionary.at<float>(j, i) = B[M*j+i];  
			}
		}
		matClose(pmatFile);  
		mxFree(B);  
		fs.open(dictionaryFile, FileStorage::WRITE);
		if (fs.isOpened())
		{
			//��vocabulary���󱣴���fs����ָ����xml�ļ���vocabulary��ǩ��
			fs << "dictionary" << dictionary;
		}
	}
	return dictionary;
}
*/

/*
 * ����ȡ��dsift�����ж�ȡdescriptors���о���
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
			for (auto fileindex = filelist.begin(); fileindex != filelist.end(); fileindex++)
			{			
				string filepath = currentCategory + '\\' + (*fileindex);

				Mat descriptors;
				FileStorage fs2(filepath, FileStorage::READ);
				if(fs2.isOpened())
				{
					fs2["dsiftFeature"] >> descriptors; //����ȡ��dsift�������ȡ������descriptors
				}
				if (allDescriptors.empty())
				{
					allDescriptors.create(0, descriptors.cols, descriptors.type()); 
				}
				allDescriptors.push_back(descriptors); //�����ж�ȡ��dsift����������ŵ�allDescriptors
			}
		}
		assert(!allDescriptors.empty());
		cout << "build dictionary..." << endl;
		BOWKMeansTrainer bowTrainer(wordCount);
		dictionary = bowTrainer.cluster(allDescriptors); //���о���

		fs.open(dictionaryFile, FileStorage::WRITE);
		if (fs.isOpened())
		{
			//��vocabulary���󱣴���fs����ָ����xml�ļ���vocabulary��ǩ��
			fs << "dictionary" << dictionary;
		}
		cout << "done build dictionary..." << endl;
	}	
	return dictionary;
}

