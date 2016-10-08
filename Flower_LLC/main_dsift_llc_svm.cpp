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
	MakeDir("result\\train\\dsiftFeature"); //���ɴ��dsift����Ŀ¼
	MakeDir("result\\train\\llcFeature"); //���ɴ��llc����Ŀ¼
	MakeDir("result\\test");
	MakeDir("result\\test\\dsiftFeature"); //���ɴ��dsift����Ŀ¼
	MakeDir("result\\test\\llcFeature"); //���ɴ��llc����Ŀ¼

	int choice;
	cout<<"1. Train\n2. Test\n"<<endl;
	cin >>choice;

	char modelFile[] = "result\\model.txt"; //ģ��

	switch(choice)
	{
	case 1:
		{
			cout<<"****************** Extracting dsift feature *******************"<<endl;	
			string databaseDir = "image\\train"; //ѵ����Ŀ¼
			string dsiftFeatureDir = "result\\train\\dsiftFeature"; //��ȡ��dsift�������Ŀ¼
			int step = 6; 
			int binSize = 5;
			int patchSize = 16;
			extractDsiftFeature(databaseDir, dsiftFeatureDir, step, binSize, patchSize); //��ȡdsift����
			cout<<"****************** Extracting dsift feature end ***************"<<endl<<endl;


			cout<<"****************** Reading dictionary *************************"<<endl;
			Mat dictionary;
			int wordCount = 1024; //�ֵ��С
			string dictionaryFile = "result\\dictionary.xml.gz"; //�ֵ���Ŀ¼
			dictionary = buildDictionary(dsiftFeatureDir, dictionaryFile, wordCount); //�����ֵ�
			//dictionary = readDictionary(dictionaryFile);
			cout<<"****************** Reading dictionary end *********************"<<endl<<endl;


			cout<<"****************** Locality-constrained Linear Coding *********"<<endl;
			int knn = 5; //number of neighbors for local coding
			string llcFeatureDir = "result\\train\\llcFeature";
			string feaTxt = "result\\train.txt";
			llc_coding_pooling(databaseDir, dsiftFeatureDir, llcFeatureDir, feaTxt, dictionary, knn);
			cout<<"****************** Locality-constrained Linear Coding end *****"<<endl<<endl;


			cout<<"****************** Liblinear flower classfication *************"<<endl;
			char trainFile[] = "result\\train.txt"; //ѵ����			
			SVM_train(argc, argv, trainFile, modelFile);
			cout<<"****************** Liblinear flower classfication end *********"<<endl;
		}
		//break;
	case 2:
		{
			cout<<"****************** Extracting dsift feature *******************"<<endl;	
			string databaseDir = "image\\test"; //ѵ����Ŀ¼
			string dsiftFeatureDir = "result\\test\\dsiftFeature"; //��ȡ��dsift�������Ŀ¼
			int step = 6; 
			int binSize = 5;
			int patchSize = 16;
			extractDsiftFeature(databaseDir, dsiftFeatureDir, step, binSize, patchSize); //��ȡdsift����
			cout<<"****************** Extracting dsift feature end ***************"<<endl<<endl;


			cout<<"****************** Reading dictionary *************************"<<endl;
			Mat dictionary;
			string dictionaryFile = "result\\dictionary.xml.gz"; //�ֵ���Ŀ¼
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
			char testFile[] = "result\\test.txt"; //���Լ�
			char resultFile[] = "result\\result.txt"; //���Խ��
			SVM_predict(argc, argv, testFile, modelFile, resultFile);
			cout<<"****************** Liblinear flower classfication end *********"<<endl;
		}
	}

	return 0;
}



