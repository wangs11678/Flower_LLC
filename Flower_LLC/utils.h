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

//生成目录
void MakeDir(const string& filepath);

//获取directory下文件夹名，存入向量dirlist
void GetDirList(const string& directory, vector<string>* dirlist);

//获取directory下文件名，存入向量filelist
void GetFileList(const string& directory, vector<string>* filelist);