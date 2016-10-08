#include "dsift.h"
#include "utils.h"

//利用vlfeat库提取dsift特征
Mat dsift(Mat img, int step, int binSize)
{
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

//生成采样点的网格矩阵
void meshgrid(const cv::Range &xgv, const cv::Range &ygv, int step, cv::Mat &X, cv::Mat &Y)  
{  
    std::vector<int> t_x, t_y;  
    for(int i = xgv.start; i <= xgv.end; i += step) t_x.push_back(i);  
    for(int j = ygv.start; j <= ygv.end; j += step) t_y.push_back(j);  
  
    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);  
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);  
} 

Mat normImg(Mat image, int maxImgSize)
{
	//将原图像转为灰度图像 
	Mat grayImg;
	cvtColor(image, grayImg, CV_BGR2GRAY); 
	image = grayImg;

	//将原图像等比例缩小到300以下
	if(image.rows > maxImgSize || image.rows >maxImgSize) 
	{	
		int max = image.rows>image.cols?image.rows:image.cols;
		double scale = double(maxImgSize)/max;
		Size dsize = Size(image.cols*scale, image.rows*scale);
		Mat destImg = Mat(dsize, CV_32S);
		resize(image, destImg, dsize);
		image = destImg;
	}
	grayImg.release();
	return image;
}

Mat calculateSiftXY(Mat dfea, int width, int height, int patchSize, int step, bool flag)
{
	Mat feaSet_x(dfea.rows, 1, CV_32FC1);
	Mat feaSet_y(dfea.rows, 1, CV_32FC1);
	int remX = (width - patchSize) % step;
	int offsetX = floor(remX/2) + 1;
	int remY = (height - patchSize) % step;
	int offsetY = floor(remY/2) + 1;

	cv::Mat gridX, gridY, gridXX, gridYY;
	meshgrid(cv::Range(offsetX, width-patchSize+1), cv::Range(offsetY, height-patchSize+1), step, gridXX, gridYY); 

	transpose(gridXX, gridX);
	transpose(gridYY, gridY);

	int gxrows = gridX.rows;
	int gxcols = gridX.cols;
	int gyrows = gridY.rows;
	int gycols = gridY.cols;

	for(int i = 0; i < gxrows; i++)
	{
		for(int j = 0; j < gxcols; j++)
		{
			feaSet_x.at<float>(j+i*gxcols, 0) = gridX.ptr<int>(i)[j] + patchSize/2 - 0.5;
		}
	}

	for(int i = 0; i < gyrows; i++)
	{
		for(int j = 0; j < gycols; j++)
		{
			feaSet_y.at<float>(j+i*gycols, 0) = gridY.ptr<int>(i)[j] + patchSize/2 - 0.5;
		}
	} 
	gridX.release();
	gridY.release();
	gridXX.release();
	gridYY.release();
	if(flag)
	{
		return feaSet_y;
	}
	return feaSet_x;
}

void extractDsiftFeature(string databaseDir, string dsiftFeatureDir, int step, int binSize, int patchSize)
{	
	int maxImgSize = 300;
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
				continue;
			} 
			else 
			{	
				string filepath = currentCategory + '\\' + *fileitr; //当前类中的所有图片名称
				Mat image = imread(filepath);

				if (image.empty())
				{
					continue; // maybe not an image file
				}	
				//预处理图片
				Mat img = normImg(image, maxImgSize);
				int width = img.cols;
				int height = img.rows;
				// dsiftDescriptor提取dsift特征
				dsiftFeature = dsift(img, step, binSize);

				Mat feaSet_x = calculateSiftXY(dsiftFeature, width, height, patchSize, step, false);
				Mat feaSet_y = calculateSiftXY(dsiftFeature, width, height, patchSize, step, true);

				fs.open(descriptorFileName, FileStorage::WRITE);
				if (fs.isOpened())
				{
					fs << "dsiftFeature" << dsiftFeature;
					fs << "feaSet_x" << feaSet_x;
					fs << "feaSet_y" << feaSet_y;
					fs << "width" << width;
					fs << "height" << height;
				}
			}
		}
	}
}