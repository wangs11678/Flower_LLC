#include "llc.h"
#include "utils.h"

cv::Mat findKNN(cv::Mat &codebook, cv::Mat &input, int knn)
{  
	int nframe = input.rows; //input(前面提取的dsift特征)矩阵的行
    int nbase = codebook.rows; //codebook(字典)矩阵的行

    Mat ii = input.mul(input); //逐元素乘法
    Mat cc = codebook.mul(codebook);  
        
    Mat sii(nframe, 1, CV_32FC1); //相当于matlab程序里的XX 
    sii.setTo(0);  
    Mat scc(nbase, 1, CV_32FC1); //相当于matlab程序里的BB 
    scc.setTo(0);  

	//将ii矩阵每一行元素加起来赋值给sii矩阵
    for (int i = 0; i < ii.rows; i++)
	{  
		for (int j = 0; j < ii.cols; j++)
		{  
			sii.at<float>(i, 0) += ii.at<float>(i, j);  
		}  
    }  
	//将cc矩阵每一行元素加起来赋值给scc矩阵
    for (int i = 0; i < cc.rows; i++)
	{  
		for (int j = 0; j < cc.cols; j++)
		{  
			scc.at<float>(i, 0) += cc.at<float>(i, j);  
		}  
    } 

    Mat D(nframe, nbase, CV_32FC1);  
    for (int i = 0; i < nframe; i++)
	{  
		for (int j = 0; j < nbase; j++)
		{  
			D.at<float>(i, j) = sii.at<float>(i, 0);  
		}  
    }   

    Mat ct;  	
    transpose(codebook, ct); //注意转置阵不能返回给Mat本身
    Mat D1 = 2*input*ct; 

    Mat scct;  
    transpose(scc, scct);  
    Mat D2(nframe, nbase, CV_32FC1);  
    for (int i = 0; i < nframe; i++)
	{  
		for (int j = 0; j < nbase; j++)
		{  
			D2.at<float>(i, j) = scct.at<float>(0, j);  
		}  
    }  
    D = D - D1 + D2;  

	Mat SD(nframe, nbase, CV_16UC1);
    sortIdx(D, SD, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING); //对D所有行升序排列后，将索引赋给SD
	//cout<<SD<<endl;

    Mat IDX(nframe, knn, CV_16UC1);  
    for (int i = 0; i < nframe; i++)
	{  
		for (int j = 0; j < knn; j++)
		{  
			//cout<<SD.at<int>(i, j)<<",,,";
			//IDX.at<uchar>(i, j) = SD.row(i).col(j).at<uchar>(0, 0);
			IDX.at<ushort>(i, j) = SD.at<int>(i, j);
			//访问CV_8UC1类型矩阵元素的一种方法ptr，此处用at会出错
			//IDX.ptr<ushort>(i)[j] = SD.ptr<int>(i)[j];	
		}  
    }   
	//cout<<IDX<<endl;
    ii.release();  
    cc.release();  
    sii.release();  
    scc.release();  
    D.release();  
    ct.release();  
    D1.release();  
    scct.release();  
    D2.release();  
    SD.release();  
    return IDX;  
}  

cv::Mat llccoding(cv::Mat &codebook, cv::Mat &input, int knn)  
{  
	//find k nearest neighbors
	Mat IDX = findKNN(codebook, input, knn);

    int nframe = input.rows;  
    int nbase = codebook.rows;  
    int dim = input.cols;  
    
    Mat II = Mat::eye(knn, knn, CV_32FC1);  
    Mat Coeff(nframe, nbase, CV_32FC1);  
    Coeff.setTo(0);  
    Mat z;  
    Mat z1(knn, dim, CV_32FC1);  
    Mat z2(knn, dim, CV_32FC1);  

    Mat C;  
    Mat un(knn, 1, CV_32FC1);  
    un.setTo(1);  
    Mat temp;  
    Mat temp2;  
    Mat w;  
    Mat wt;  
      
    for (int n = 0; n < nframe; n++)
	{  
        for (int i = 0; i < knn; i++)
		{  
            for (int j = 0; j < dim; j++)
			{  
                z1.at<float>(i, j) = codebook.at<float>(IDX.ptr<uchar>(n)[i], j);  
                z2.at<float>(i, j) = input.at<float>(n, j);  
            }  
        }  
        z = z1 - z2;  
        transpose(z, temp);  
        C = z*temp;  
        C = C + II*(1e-4)*trace(C)[0];  
        invert(C, temp2);  
        w = temp2*un;  
        float sum_w = 0;  
        for (int i = 0; i < knn; i++)
		{  
            sum_w += w.at<float>(i, 0);  
        }  
        w = w/sum_w;  
        transpose(w, wt);  
        for (int i = 0; i < knn; i++)
		{  
            Coeff.at<float>(n, IDX.ptr<uchar>(n)[i]) = wt.at<float>(0, i);  
        }  
    }  
      
    II.release();  
    z.release();  
    z1.release();  
    z2.release();  
    C.release();  
    un.release();  
    temp.release();  
    temp2.release();  
    w.release();  
    wt.release();  
      
    return Coeff;  
}   

cv::Mat llcpooling(cv::Mat &tcodebook, cv::Mat &tinput, int knn, cv::Mat llccode, string imageFileName)
{
	Mat codebook, input;
	transpose(tcodebook, codebook);
	transpose(tinput, input);

	int dSize = codebook.cols;
	int nSmp = input.cols;
	Mat image = imread(imageFileName);
	int img_width = image.rows;
	int img_height = image.cols;
	Mat idxBin(nSmp, 1, CV_32FC1);
	Mat llccodes;
	transpose(llccode, llccodes); 

	Mat pyramid(1,3,CV_32FC1);
	pyramid.at<float>(0, 0) = 1;
	pyramid.at<float>(0, 1) = 2;
	pyramid.at<float>(0, 2) = 4;

	int pLevels = pyramid.cols;
	Mat pBins(1,3,CV_32FC1);
	pBins.at<float>(0, 0) = pyramid.at<float>(0, 0)*pyramid.at<float>(0, 0);
	pBins.at<float>(0, 1) = pyramid.at<float>(0, 1)*pyramid.at<float>(0, 1);
	pBins.at<float>(0, 2) = pyramid.at<float>(0, 2)*pyramid.at<float>(0, 2);

	int tBins = pBins.at<float>(0, 0) + pBins.at<float>(0, 1) + pBins.at<float>(0, 2);
	Mat beta(dSize, tBins, CV_32FC1);
	int bId = 0;

	for (int iter1 = 0; iter1 != pLevels; iter1++)
	{
		int nBins = pBins.at<float>(0, iter1); 
		float wUnit = img_width / pyramid.at<float>(0, iter1);
		float hUnit = img_height / pyramid.at<float>(0, iter1);

		Mat feaSet_x(tinput.cols, 1, CV_32FC1);
		Mat feaSet_y(tinput.cols, 1, CV_32FC1);

		//find to which spatial bin each local descriptor belongs
		//int xBin = cvCeil(feaSet.x / wUnit);
		//int yBin = cvCeil(feaSet.y / hUnit);

		//idxBin = (yBin - 1)*pyramid(iter1) + xBin;
	}

	return beta;
}

void llc_coding_pooling(string databaseDir, string dsiftFeatureDir, string llcFeatureDir, cv::Mat &codebook, int knn)
{
	vector<string> categories;
	GetDirList(databaseDir, &categories);

	Mat dsiftFeature, llccodes, llcFeature;

	for (int index = 0; index != categories.size(); index++)
	{
		string currentCategoryDatabase = databaseDir + "\\" + categories[index];
		string currentCategoryDsift = dsiftFeatureDir + '\\' + categories[index];
		vector<string> filelist;
		GetFileList(currentCategoryDatabase, &filelist);

		for (auto fileindex = filelist.begin(); fileindex != filelist.end(); fileindex++)
		{		
			string llcFileName = llcFeatureDir + "\\" + categories[index] + "\\" + (*fileindex) + ".xml.gz";
			MakeDir(llcFeatureDir + "\\" + categories[index]);
			FileStorage fs(llcFileName, FileStorage::READ);
			if (fs.isOpened())
			{ 
				// already cached
				//fs["llcFeature"] >> llcFeature;
			}			
			else
			{
				string dsiftFileName = currentCategoryDsift + "\\" + (*fileindex) + ".xml.gz";
				FileStorage fs2(dsiftFileName, FileStorage::READ);	
				if (fs2.isOpened())
				{ 
					// already cached
					fs2["dsiftFeature"] >> dsiftFeature;
				}

				string imageFileName = currentCategoryDatabase + "\\" + (*fileindex);
				llccodes = llccoding(codebook, dsiftFeature, knn);
				//llcFeature = llcpooling(codebook, dsiftFeature, knn, llccodes, imageFileName);

				fs.open(llcFileName, FileStorage::WRITE);
				if (fs.isOpened())
				{
					fs << "llcFeature" << llcFeature;
				}
			}			
		}
	}
}