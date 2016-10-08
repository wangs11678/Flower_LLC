#include "dsift.h"
#include "vocabulary.h"
#include "llc.h"
#include "train.h"
#include "predict.h"
#include "utils.h"

int main(int argc, char* argv[])
{
	MakeDir("result");
	MakeDir("result\\train");
	MakeDir("result\\train\\dsiftFeature"); //生成存放dsift特征目录
	MakeDir("result\\train\\llcFeature"); //生成存放llc特征目录
	MakeDir("result\\test");
	MakeDir("result\\test\\dsiftFeature"); //生成存放dsift特征目录
	MakeDir("result\\test\\llcFeature"); //生成存放llc特征目录

	int choice;
	cout<<"1. Train\n2. Test\n"<<endl;
	cin >>choice;

	char modelFile[] = "result\\model.txt"; //模型

	switch(choice)
	{
	case 1:
		{
			cout<<"****************** Extracting dsift feature *******************"<<endl;	
			string databaseDir = "image\\train"; //训练集目录
			string dsiftFeatureDir = "result\\train\\dsiftFeature"; //提取的dsift特征存放目录
			int step = 6; 
			int binSize = 5;
			int patchSize = 16;
			extractDsiftFeature(databaseDir, dsiftFeatureDir, step, binSize, patchSize); //提取dsift特征
			cout<<"****************** Extracting dsift feature end ***************"<<endl<<endl;


			cout<<"****************** Reading dictionary *************************"<<endl;
			Mat dictionary;
			int wordCount = 1024; //字典大小
			string dictionaryFile = "result\\dictionary.xml.gz"; //字典存放目录
			dictionary = buildDictionary(dsiftFeatureDir, dictionaryFile, wordCount); //构造字典
			//dictionary = readDictionary(dictionaryFile);
			cout<<"****************** Reading dictionary end *********************"<<endl<<endl;


			cout<<"****************** Locality-constrained Linear Coding *********"<<endl;
			int knn = 5; //number of neighbors for local coding
			string llcFeatureDir = "result\\train\\llcFeature";
			string feaTxt = "result\\train.txt";
			llc_coding_pooling(databaseDir, dsiftFeatureDir, llcFeatureDir, feaTxt, dictionary, knn);
			cout<<"****************** Locality-constrained Linear Coding end *****"<<endl<<endl;


			cout<<"****************** Liblinear flower classfication *************"<<endl;
			char trainFile[] = "result\\train.txt"; //训练集			
			SVM_train(argc, argv, trainFile, modelFile);
			cout<<"****************** Liblinear flower classfication end *********"<<endl;
		}
		//break;
	case 2:
		{
			cout<<"****************** Extracting dsift feature *******************"<<endl;	
			string databaseDir = "image\\test"; //训练集目录
			string dsiftFeatureDir = "result\\test\\dsiftFeature"; //提取的dsift特征存放目录
			int step = 6; 
			int binSize = 5;
			int patchSize = 16;
			extractDsiftFeature(databaseDir, dsiftFeatureDir, step, binSize, patchSize); //提取dsift特征
			cout<<"****************** Extracting dsift feature end ***************"<<endl<<endl;


			cout<<"****************** Reading dictionary *************************"<<endl;
			Mat dictionary;
			string dictionaryFile = "result\\dictionary.xml.gz"; //字典存放目录
			//dictionary = readDictionary(dictionaryFile);
			FileStorage fs(dictionaryFile, FileStorage::READ);
			if(fs.isOpened())
			{
				fs["dictionary"] >> dictionary;
			}
			cout<<"****************** Reading dictionary end *********************"<<endl<<endl;


			cout<<"****************** Locality-constrained Linear Coding *********"<<endl;
			int knn = 5; //number of neighbors for local coding
			string llcFeatureDir = "result\\test\\llcFeature";
			string feaTxt = "result\\test.txt";
			llc_coding_pooling(databaseDir, dsiftFeatureDir, llcFeatureDir, feaTxt, dictionary, knn);
			cout<<"****************** Locality-constrained Linear Coding end *****"<<endl<<endl;


			cout<<"****************** Liblinear flower classfication *************"<<endl;
			char testFile[] = "result\\test.txt"; //测试集
			char resultFile[] = "result\\result.txt"; //测试结果
			SVM_predict(argc, argv, testFile, modelFile, resultFile);
			cout<<"****************** Liblinear flower classfication end *********"<<endl;
		}
	}

	return 0;
}



