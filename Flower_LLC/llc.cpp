#include "llc.h"
#include "utils.h"

//find k nearest neighbors
cv::Mat findKNN(cv::Mat &B, cv::Mat &X, int knn)
{  
	int nframe = X.rows; //X(前面提取的dsift特征)矩阵的行
    int nbase = B.rows; //B(字典)矩阵的行
     
	Mat XX, BB;
	//reduce矩阵变向量，相当于matlab中的sum
	cv::reduce(X.mul(X), XX, 1, CV_REDUCE_SUM, CV_32FC1);
	cv::reduce(B.mul(B), BB, 1, CV_REDUCE_SUM, CV_32FC1);

	//repeat相当于matlab中的repmat
	Mat D1 = cv::repeat(XX, 1, nbase);
    Mat Bt;  	
    transpose(B, Bt); //注意转置阵不能返回给原Mat本身
    Mat D2 = 2*X*Bt; 
    Mat BBt;  
    transpose(BB, BBt);  
	Mat D3 = cv::repeat(BBt, nframe, 1);
    Mat D = D1 - D2 + D3;  

	Mat SD(nframe, nbase, CV_16UC1);
    cv::sortIdx(D, SD, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING); //对D所有行升序排列后，将索引赋给SD

    Mat IDX(nframe, knn, CV_16UC1);  
	//将SD的d第i列赋值给IDX的第i列
	for (int i = 0; i < knn; i++)
	{
		SD.col(i).copyTo(IDX.col(i));
	}
	//cout<<IDX<<endl;

    XX.release();
    BB.release();
	Bt.release();
	BBt.release();
    D.release();    
    D1.release();    
    D2.release(); 
	D3.release();
    SD.release();  
    return IDX;  
}  

//llc approximation coding
cv::Mat llccoding(cv::Mat &B, cv::Mat &X, int knn)  
{  
	float beta = 1e-4;
	//find k nearest neighbors
	Mat IDX = findKNN(B, X, knn);

    int nframe = X.rows; //特征行 
	int nxcol = X.cols; //特征列
    int nbase = B.rows; //字典行 
    
    Mat II = Mat::eye(knn, knn, CV_32FC1);  
	Mat Coeff = Mat::zeros(nframe, nbase, CV_32FC1);  

    Mat z, zt;  
    Mat z1(knn, nxcol, CV_32FC1);  
    Mat z2(knn, nxcol, CV_32FC1);  

    Mat C, C_inv;   
    Mat w, wt;   
      
    for (int i = 0; i < nframe; i++)
	{  
        for (int j = 0; j < knn; j++)
		{  
			B.row(IDX.ptr<ushort>(i)[j]).copyTo(z1.row(j));
			X.row(i).copyTo(z2.row(j)); 
        }  
        z = z1 - z2; 

        transpose(z, zt);  
        C = z*zt;  
        C = C + II*beta*trace(C)[0]; //trace(C)[0]求矩阵的迹
        invert(C, C_inv);  

        w = C_inv*Mat::ones(knn, 1, CV_32FC1); //相当于matlab中的w = C\ones(knn,1);

        float sum_w = 0; 
		sum_w = cv::sum(w)[0];
        w = w/sum_w; 
        transpose(w, wt);  

        for (int j = 0; j < knn; j++)
		{  
            Coeff.at<float>(i, IDX.ptr<ushort>(i)[j]) = wt.at<float>(0, j);  
        }  
    }  
	//cout<<Coeff;

    II.release();  
    z.release(); 
	zt.release(); 
    z1.release();  
    z2.release();  
    C.release();  
	C_inv.release();     
    w.release();  
    wt.release();     
    return Coeff;  
}   

cv::Mat llcpooling(cv::Mat &tcodebook,
				   cv::Mat &tinput,
				   cv::Mat feaSet_x, 
				   cv::Mat feaSet_y, 
				   int width,
				   int height,
				   cv::Mat tllccodes)
{
	Mat codebook, input;
	transpose(tcodebook, codebook);
	transpose(tinput, input);

	int dSize = codebook.cols;
	int nSmp = input.cols;

	Mat idxBin = Mat::zeros(nSmp, 1, CV_32FC1);
	Mat llccodes;
	transpose(tllccodes, llccodes); 

	Mat pyramid(1, 3, CV_32FC1);
	pyramid.at<float>(0, 0) = 1;
	pyramid.at<float>(0, 1) = 2;
	pyramid.at<float>(0, 2) = 4;

	int pLevels = pyramid.cols;
	Mat pBins(1,3,CV_32FC1);
	int tBins = 0;
	for(int i = 0; i < 3; i++)
	{
		pBins.at<float>(0, i) = pyramid.at<float>(0, i)*pyramid.at<float>(0, i);
		tBins += pBins.at<float>(0, i);
	}
	Mat beta = Mat::zeros(dSize, tBins, CV_32FC1);
	int bId = 0;
	int betacol = -1; //beta的列


	for (int iter1 = 0; iter1 != pLevels; iter1++)
	{
		int nBins = pBins.at<float>(0, iter1); 
		float wUnit = width / pyramid.at<float>(0, iter1);
		float hUnit = height / pyramid.at<float>(0, iter1);   

		//find to which spatial bin each local descriptor belongs
		Mat xBin(nSmp, 1, CV_32FC1);
		Mat yBin(nSmp, 1, CV_32FC1);

		for(int i = 0; i < nSmp; i++)
		{
			xBin.at<float>(i, 0) = ceil(feaSet_x.at<float>(i, 0) / wUnit);
		    yBin.at<float>(i, 0) = ceil(feaSet_y.at<float>(i, 0) / hUnit);
			idxBin.at<float>(i, 0) = (yBin.at<float>(i, 0) - 1) * pyramid.at<float>(0, iter1) + xBin.at<float>(i, 0);
		}	

		for(int iter2 = 1; iter2 <= nBins; iter2++)
		{
			bId = bId + 1;
			betacol = betacol + 1;

			int nsbrows = 0; //统计每次循环sidxBin的行总数
			for(int i = 0; i < nSmp; i++)
			{		
				if(idxBin.at<float>(i, 0) == iter2)
				{
					nsbrows++;
				}
			}

			Mat sidxBin(nsbrows, 1, CV_16UC1);
			int sbrow = 0; //sidxBin的行
			for(int i = 0; i < nSmp; i++)
			{
				if(idxBin.at<float>(i, 0) == iter2)
				{
					sidxBin.ptr<ushort>(sbrow++)[0] = i;
				}
			}
			if(sidxBin.empty())
			{
				continue;
			}	
			//cout<<sidxBin;

			//beta(:, bId) = max(llc_codes(:, sidxBin), [], 2);
			float iRowMax = 0; //每一行的最大值
			for(int i = 0; i < llccodes.rows; i++)
			{
				iRowMax = llccodes.at<float>(i, sidxBin.ptr<ushort>(0)[0]);
				for(int j = 0; j < nsbrows; j++)
				{
					if(llccodes.at<float>(i, sidxBin.ptr<ushort>(j)[0]) > iRowMax)
					{
						iRowMax = llccodes.at<float>(i, sidxBin.ptr<ushort>(j)[0]);
					}
				}
				beta.at<float>(i, betacol) = iRowMax;
			}
		}
	}
	//cout<<beta<<endl;

	if(bId != tBins)
	{
		cout<<"Index number error!"<<endl;
		exit;
	}

	Mat fea(dSize*tBins, 1, CV_32FC1);
	for(int i = 0; i < dSize; i++)
	{
		for(int j = 0; j < tBins; j++)
		{
			fea.at<float>(j+i*tBins, 0) = beta.at<float>(i, j);
		}
	}

	float sum = 0; //注意类型是float不是int
	for(int i = 0; i < dSize*tBins; i++)
	{
		sum += fea.at<float>(i, 0) * fea.at<float>(i, 0);
	}
	fea = fea/sqrt(sum);
	//cout<<fea;

	codebook.release();
	input.release();
	idxBin.release();
	idxBin.release();
	llccodes.release();
	pyramid.release();
	pBins.release();
	beta.release();
	feaSet_x.release();
	feaSet_y.release();

	return fea;
}

void llc_coding_pooling(string databaseDir, 
						string dsiftFeatureDir, 
						string llcFeatureDir, 
						string feaTxt,
						cv::Mat &codebook, 
						int knn)
{
	int width, height;
	Mat dsiftFeature, feaSet_x, feaSet_y, llccodes, llcFeature;

	vector<string> categories;
	GetDirList(databaseDir, &categories);

	ofstream outfile(feaTxt);  

	for (int index = 0; index != categories.size(); index++)
	{
		cout<<"Coding and pooling the of class "<<categories[index]<<"..."<<endl;
		string currentCategoryDatabase = databaseDir + "\\" + categories[index];
		string currentCategoryDsift = dsiftFeatureDir + '\\' + categories[index];
		vector<string> filelist;
		GetFileList(currentCategoryDatabase, &filelist);

		for (auto fileindex = filelist.begin(); fileindex != filelist.end(); fileindex++)
		{	
			outfile<<index+1<<"\t";
			string llcFileName = llcFeatureDir + "\\" + categories[index] + "\\" + (*fileindex) + ".xml.gz";
			MakeDir(llcFeatureDir + "\\" + categories[index]);
			FileStorage fs(llcFileName, FileStorage::READ);
			if (fs.isOpened())
			{ 
				// already cached
				fs["llcFeature"] >> llcFeature;
				for(int i = 0; i < llcFeature.rows; i++)
				{
					outfile<<i+1<<":"<<llcFeature.at<float>(i, 0)<<"\t";
				}
			}			
			else
			{
				string dsiftFileName = currentCategoryDsift + "\\" + (*fileindex) + ".xml.gz";
				FileStorage fs2(dsiftFileName, FileStorage::READ);	
				if (fs2.isOpened())
				{ 
					// already cached
					fs2["dsiftFeature"] >> dsiftFeature;
					fs2["feaSet_x"] >> feaSet_x;
					fs2["feaSet_y"] >> feaSet_y;
					fs2["width"] >> width;
					fs2["height"] >> height;
				}
				llccodes = llccoding(codebook, dsiftFeature, knn);
				llcFeature = llcpooling(codebook, dsiftFeature, feaSet_x, feaSet_y, width, height, llccodes);

				fs.open(llcFileName, FileStorage::WRITE);
				if (fs.isOpened())
				{
					fs << "llcFeature" << llcFeature;
					for(int i = 0; i < llcFeature.rows; i++)
					{
						outfile<<i+1<<":"<<llcFeature.at<float>(i, 0)<<"\t";
					}
				}
			}
			outfile<<"\n";
		}
	}
	dsiftFeature.release();
	feaSet_x.release();
	feaSet_y.release();
	llccodes.release();
	llcFeature.release();
}