// windows api
#include <Windows.h>
#include <tchar.h>
#include <strsafe.h>
#pragma comment( lib, "User32.lib")
// c api
#include <stdio.h>
#include <string.h>
#include <assert.h>
// c++ api
#include <string>
#include <map>
#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>

using namespace std;

//����Ŀ¼
void MakeDir(const string& filepath);

//��ȡdirectory���ļ���������������dirlist
void GetDirList(const string& directory, vector<string>* dirlist);

//��ȡdirectory���ļ�������������filelist
void GetFileList(const string& directory, vector<string>* filelist);