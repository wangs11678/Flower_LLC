#include "dsift.h"
#include "dictionary.h"
#include "llc.h"
#include "utils.h"

int main(int argc, char* argv[])
{
	MakeDir("result");
	MakeDir("result\\dsiftFeature"); //生成存放dsift特征目录
	MakeDir("result\\llcFeature"); //生成存放llc特征目录

	/****************** Extracting dsift Feature *************************************************/	
	cout<<"****************** Extracting dsift feature *******************"<<endl;	
	string databaseDir = "data\\train"; //训练集目录
	string dsiftFeatureDir = "result\\dsiftFeature"; //提取的dsift特征存放目录
	int step = 6; 
	int binSize = 5;
	//extractDsiftFeature(databaseDir, dsiftFeatureDir, step, binSize); //提取dsift特征
	cout<<"****************** Extracting dsift feature end ***************"<<endl<<endl;
	

	/****************** Building Dictionary ******************************************************/
	cout<<"****************** Building dictionary ************************"<<endl;
	Mat dictionary;
	int wordCount = 1024; //字典大小
	string dictionaryFile = "result\\dictionary.xml.gz"; //字典存放目录
	dictionary = buildDictionary(dsiftFeatureDir, dictionaryFile, wordCount); //构造字典
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



