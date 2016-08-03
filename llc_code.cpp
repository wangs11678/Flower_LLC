#include "llc_code.h"
#include "utils.h"

cv::Mat findKNN(cv::Mat &codebook, cv::Mat &input, int k) {  
    int nbase = codebook.rows;  
    int nquery = input.rows;  
    Mat ii = input.mul(input);  
    Mat cc = codebook.mul(codebook);  
        
    Mat sii(nquery,1,CV_32FC1);  
    sii.setTo(0);  
    Mat scc(nbase,1,CV_32FC1);  
    scc.setTo(0);  
    for (int i = 0; i < ii.rows; i++)
	{  
		for (int j = 0; j < ii.cols; j++)
		{  
			sii.at<float>(i,0) += ii.at<float>(i,j);  
		}  
    }  
	cout<<"test1"<<endl;
    for (int i = 0; i < cc.rows; i++)
	{  
		for (int j = 0; j < cc.cols; j++)
		{  
			scc.at<float>(i,0) += cc.at<float>(i,j);  
		}  
    }  
    cout<<"test2"<<endl;    
    Mat D(nquery, nbase, CV_32FC1);  
    for (int i = 0; i < nquery; i++)
	{  
		for (int j = 0; j < nbase; j++)
		{  
			D.at<float>(i,j) = sii.at<float>(i,0);  
		}  
    }  
    cout<<"test3"<<endl;  
    Mat ct;  
	//注意转置阵不能返回给Mat本身
    transpose(codebook, ct); 

    Mat D1 = 2*input*ct;  
    cout<<"test4"<<endl;    
    Mat scct;  
    transpose(scc, scct);  
    Mat D2(nquery, nbase, CV_32FC1);  
	cout<<"test5"<<endl; 
    for (int i = 0; i < nquery; i++)
	{  
		for (int j = 0; j < nbase; j++)
		{  
			D2.at<float>(i,j) = scct.at<float>(0,j);  
		}  
    }  
	cout<<"test6"<<endl; 
    D = D - D1 + D2;  
    Mat SD;  
    sortIdx(D, SD, CV_SORT_EVERY_ROW+CV_SORT_ASCENDING);  
    Mat IDX(nquery, k, CV_8UC1);  
	cout<<"test7"<<endl;
    for (int i = 0; i < nquery; i++)
	{  
		for (int j = 0; j < k; j++)
		{  
			IDX.at<uchar>(i,j) = SD.row(i).col(j).at<uchar>(0,0);  
		}  
    }   
	cout<<"test8"<<endl;
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
  

cv::Mat llccode(cv::Mat &codebook, cv::Mat &input, cv::Mat IDX, int k)  
{  
    int nquery = input.rows;  
    int nbase = codebook.rows;  
    int dim = codebook.cols;  
      
    Mat II = Mat::eye(k, k, CV_32FC1);  
    Mat Coeff(nquery, nbase, CV_32FC1);  
    Coeff.setTo(0);  
    Mat z;  
    Mat z1(k, dim, CV_32FC1);  
    Mat z2(k, dim, CV_32FC1);  
    Mat C;  
    Mat un(k, 1, CV_32FC1);  
    un.setTo(1);  
    Mat temp;  
    Mat temp2;  
    Mat w;  
    Mat wt;  
      
    for (int n = 0; n < nquery; n++)
	{  
        for (int i = 0; i < k; i++)
		{  
            for (int j = 0; j < dim; j++)
			{  
                z1.at<float>(i,j) = codebook.at<float>(IDX.at<uchar>(n,i),j);  
                z2.at<float>(i,j) = input.at<float>(n,j);  
            }  
        }  
        z = z1 - z2;  
        transpose(z, temp);  
        C = z*temp;  
        C = C + II*(1e-4)*trace(C)[0];  
        invert(C, temp2);  
        w = temp2*un;  
        float sum_w = 0;  
        for (int i = 0; i < k; i++)
		{  
            sum_w += w.at<float>(i, 0);  
        }  
        w = w/sum_w;  
        transpose(w, wt);  
        for (int i = 0; i < k; i++)
		{  
            Coeff.at<float>(n,IDX.at<uchar>(n,i)) = wt.at<float>(0,i);  
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

void llc(cv::Mat &codebook, int k)
{
	string siftDir("result\\sift");
	string llcDir("result\\llc");
	MakeDir("result/llc");

	vector<string> categories;
	GetDirList(siftDir, &categories);

	Mat llcFea, siftFea, idx;

	for (int index = 0; index != categories.size(); index++)
	{
		string currentCategory = siftDir + '\\' + categories[index];
		vector<string> filelist;
		GetFileList(currentCategory, &filelist);

		for (auto fileindex = filelist.begin(); fileindex != filelist.end(); fileindex++)
		{		
			string llcFileName = llcDir + "\\" + categories[index] + "\\" + (*fileindex) + ".xml.gz";
			MakeDir("result/llc/" + categories[index]);
			FileStorage fs(llcFileName, FileStorage::READ);
			if (fs.isOpened())
			{ 
				// already cached
				fs["llcFea"] >> llcFea;
			}			
			else
			{
				string siftFileName = siftDir + "\\" + categories[index] + "\\" + (*fileindex);
				cout<<siftFileName<<endl;
				FileStorage sift_fs(siftFileName, FileStorage::READ);	
				if (sift_fs.isOpened())
				{ 
					// already cached
					sift_fs["imageDescriptor"] >> siftFea;
				}
				cout<<"llc begin"<<endl;
				idx = findKNN(codebook, siftFea, k);
				cout<<"llc ing"<<endl;
				llcFea = llccode(codebook, siftFea, idx, k);
				cout<<"llc end"<<endl;

				fs.open(llcFileName, FileStorage::WRITE);
				if (fs.isOpened())
				{
					fs << "llcFea" << llcFea;
				}
			}			
		}
	}
}