#include "dsift.h"
#include "utils.h"

//����vlfeat����ȡdsift����
Mat dsift(Mat img, int step, int binSize)
{
	//��ԭͼ��תΪ�Ҷ�ͼ�� 
	Mat grayImg;
	cvtColor(img, grayImg, CV_BGR2GRAY); 
	img = grayImg;

	//��ԭͼ��ȱ�����С��300����
	int maxImgSize = 300;
	if(img.rows > maxImgSize || img.rows >maxImgSize) 
	{	
		int max = img.rows>img.cols?img.rows:img.cols;
		float scale = float(maxImgSize)/max;
		Size dsize = Size(img.cols*scale, img.rows*scale);
		Mat destImg = Mat(dsize, CV_32S);
		resize(img, destImg, dsize);
		img = destImg;
	}

	VlDsiftFilter * vlf = vl_dsift_new_basic(img.rows, img.cols, step, binSize);

	// transform image in cv::Mat to float vector
	std::vector<float> imgvec;

	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{
			imgvec.push_back(img.at<unsigned char>(i,j) / 255.0f);                                                                                                                                                                                                        
		}
	}
	// call processing function of vl
	vl_dsift_process(vlf, &imgvec[0]);

	// echo number of keypoints found
	//����Mat�����ȡ����dsift����
	Mat dsiftDescriptor(vl_dsift_get_keypoint_num(vlf), vlf->descrSize, CV_32FC1); 

	for(int i = 0; i < vl_dsift_get_keypoint_num(vlf); i++)
	{
		for(int j = 0; j < vlf->descrSize; j++)
		{
			//����ȡ����������ֵ��dsiftDescriptor����
			dsiftDescriptor.at<float>(i,j) = vlf->descrs[j+i*vlf->descrSize]; 
		}
	}
	return dsiftDescriptor;
}

void extractDsiftFeature(string databaseDir, string dsiftFeatureDir, int step, int binSize)
{	
	vector<string> categories;
	GetDirList(databaseDir, &categories);
	for (int i = 0; i != categories.size(); i++) //����ѵ�����л�������
	{
		cout<<"Extracting the dsift of class "<<categories[i]<<"..."<<endl;
		string currentCategory = databaseDir + '\\' + categories[i]; //��ǰ��Ŀ¼
		vector<string> filelist;
		GetFileList(currentCategory, &filelist); //��ȡ��ǰ��������ͼƬ����filelist

		//������ǰ�������и��໨��ͼƬ
		for (auto fileitr = filelist.begin(); fileitr != filelist.end(); fileitr++) 
		{		
			MakeDir(dsiftFeatureDir + "\\" + categories[i]);
			//dsift�����ļ�������ʽ
			string descriptorFileName = dsiftFeatureDir + "\\" + categories[i] + "\\" + (*fileitr) + ".xml.gz"; 		
			FileStorage fs(descriptorFileName, FileStorage::READ);
			Mat dsiftFeature;
			if (fs.isOpened()) //dsift�����Ѿ�cached��ֱ�Ӷ�ȡ
			{ 
				// already cached
				//fs["dsiftDescriptor"] >> dsiftDescriptor;
			} 
			else 
			{			
				string filepath = currentCategory + '\\' + *fileitr; //��ǰ���е�����ͼƬ����
				Mat image = imread(filepath);
				if (image.empty())
				{
					continue; // maybe not an image file
				}

				// dsiftDescriptor��ȡdsift����
				dsiftFeature = dsift(image, step, binSize);

				fs.open(descriptorFileName, FileStorage::WRITE);
				if (fs.isOpened())
				{
					fs << "dsiftFeature" << dsiftFeature;
				}
			}
		}
	}
}