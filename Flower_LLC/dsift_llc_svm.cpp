#include "dsift.h"
#include "dictionary.h"
#include "llc.h"
#include "utils.h"

int main(int argc, char* argv[])
{
	MakeDir("result");
	MakeDir("result\\dsiftFeature"); //���ɴ��dsift����Ŀ¼
	MakeDir("result\\llcFeature"); //���ɴ��llc����Ŀ¼

	/****************** Extracting dsift Feature *************************************************/	
	cout<<"****************** Extracting dsift feature *******************"<<endl;	
	string databaseDir = "data\\train"; //ѵ����Ŀ¼
	string dsiftFeatureDir = "result\\dsiftFeature"; //��ȡ��dsift�������Ŀ¼
	int step = 6; 
	int binSize = 5;
	//extractDsiftFeature(databaseDir, dsiftFeatureDir, step, binSize); //��ȡdsift����
	cout<<"****************** Extracting dsift feature end ***************"<<endl<<endl;
	

	/****************** Building Dictionary ******************************************************/
	cout<<"****************** Building dictionary ************************"<<endl;
	Mat dictionary;
	int wordCount = 1024; //�ֵ��С
	string dictionaryFile = "result\\dictionary.xml.gz"; //�ֵ���Ŀ¼
	dictionary = buildDictionary(dsiftFeatureDir, dictionaryFile, wordCount); //�����ֵ�
	cout<<"****************** Building dictionary end ********************"<<endl<<endl;


	/****************** Locality-constrained Linear Coding ***************************************/
	cout<<"****************** Locality-constrained Linear Coding *********"<<endl;
	int knn = 5; //number of neighbors for local coding
	string llcFeatureDir = "result\\llcFeature";
	llc_coding_pooling(databaseDir, dsiftFeatureDir, llcFeatureDir, dictionary, knn);
	cout<<"****************** Locality-constrained Linear Coding end *****"<<endl<<endl;
	

	/****************** Liblinear flower classfication *******************************************/
	cout<<"****************** Liblinear flower classfication *************"<<endl;
	

	cout<<"****************** Liblinear flower classfication end *********"<<endl;

	return 0;
}



