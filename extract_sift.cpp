#include "extract_sift.h"
#include "utils.h"

void ExtractSIFTFeature(const string& databaseDir,
				 const vector<string>& categories, 
				 const SiftDescriptorExtractor detector,
				 const string& imageDescriptorsDir)
{	
	for (auto i = 0; i != categories.size(); ++i)
	{
		cout<<"Extracting the sift of class "<<categories[i]<<endl;
		string currentCategory = databaseDir + '\\' + categories[i];
		vector<string> filelist;
		GetFileList(currentCategory, &filelist);			
		for (auto fileitr = filelist.begin(); fileitr != filelist.end(); ++fileitr)
		{
			string descriptorFileName = imageDescriptorsDir + "\\" + categories[i] + "\\" + (*fileitr) + ".xml.gz";
			MakeDir(imageDescriptorsDir + "/" + categories[i]);
			FileStorage fs(descriptorFileName, FileStorage::READ);
			Mat imageDescriptor;
			if (fs.isOpened())
			{ 
				// already cached
				fs["imageDescriptor"] >> imageDescriptor;
			} 
			else
			{				
				string filepath = currentCategory + '\\' + *fileitr;
				Mat image = imread(filepath);
				if (image.empty())
				{
					continue; // maybe not an image file
				}
				vector<KeyPoint> keyPoints;
				detector.detect(image, keyPoints);
				detector.compute(image, keyPoints,imageDescriptor);
				//imageDescriptor = imageDescriptor.t(); //×ªÖÃ
				fs.open(descriptorFileName, FileStorage::WRITE);
				if (fs.isOpened())
				{
					fs << "imageDescriptor" << imageDescriptor;
				}
			}
		}
	}
}