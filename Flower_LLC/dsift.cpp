#include "dsift.h"
#include "utils.h"

//利用vlfeat库提取dsift特征
Mat dsift(Mat img, int step, int binSize)
{
	//将原图像转为灰度图像 
	Mat grayImg;
	cvtColor(img, grayImg, CV_BGR2GRAY); 
	img = grayImg;

	//将原图像等比例缩小到300以下
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
	//定义Mat存放提取到的dsift特征
	Mat dsiftDescriptor(vl_dsift_get_keypoint_num(vlf), vlf->descrSize, CV_32FC1); 

	for(int i = 0; i < vl_dsift_get_keypoint_num(vlf); i++)
	{
		for(int j = 0; j < vlf->descrSize; j++)
		{
			//将提取到的特征赋值给dsiftDescriptor保存
			dsiftDescriptor.at<float>(i,j) = vlf->descrs[j+i*vlf->descrSize]; 
		}
	}
	return dsiftDescriptor;
}

void extractDsiftFeature(string databaseDir, string dsiftFeatureDir, int step, int binSize)
{	
	vector<string> categories;
	GetDirList(databaseDir, &categories);
	for (int i = 0; i != categories.size(); i++) //遍历训练集中花的种类
	{
		cout<<"Extracting the dsift of class "<<categories[i]<<"..."<<endl;
		string currentCategory = databaseDir + '\\' + categories[i]; //当前类目录
		vector<string> filelist;
		GetFileList(currentCategory, &filelist); //获取当前类中所有图片存入filelist

		//遍历当前类中所有该类花的图片
		for (auto fileitr = filelist.begin(); fileitr != filelist.end(); fileitr++) 
		{		
			MakeDir(dsiftFeatureDir + "\\" + categories[i]);
			//dsift特征文件保存形式
			string descriptorFileName = dsiftFeatureDir + "\\" + categories[i] + "\\" + (*fileitr) + ".xml.gz"; 		
			FileStorage fs(descriptorFileName, FileStorage::READ);
			Mat dsiftFeature;
			if (fs.isOpened()) //dsift特征已经cached，直接读取
			{ 
				// already cached
				//fs["dsiftDescriptor"] >> dsiftDescriptor;
			} 
			else 
			{			
				string filepath = currentCategory + '\\' + *fileitr; //当前类中的所有图片名称
				Mat image = imread(filepath);
				if (image.empty())
				{
					continue; // maybe not an image file
				}

				// dsiftDescriptor提取dsift特征
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