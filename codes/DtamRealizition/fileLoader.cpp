// loads all files of a given name and extension
//#include "stdafx.h"     // windows precompiled header file
#include <iostream>
#include <set>
#include <string>
#include "fileLoader.hpp"
#include <boost/filesystem.hpp>
#include <fstream>
//#include <comdef.h>	    // part of MS Platform SDK

    using namespace cv;
    using namespace std;
    namespace fs = ::boost::filesystem;
    static fs::path root;
static vector<fs::path> txt;
static vector<fs::path> png;
static vector<fs::path> depth;


void get_all(const fs::path& root, const string& ext, vector<fs::path>& ret)
/*{
	if (!fs::exists(root) || !fs::is_directory(root))
		return;
	fs::recursive_directory_iterator it(root);
	fs::recursive_directory_iterator endit;

	while (it!=endit)
	{
		if (fs::is_regular_file(*it) && it->path().extension() == ext)
			ret.push_back(it->path().filename());
		++it;
	}
}*/;
void loadAhanda(const char * rootpath,
                double range,
                int imageNumber,
                Mat& image,
                Mat& d,
                Mat& cameraMatrix,
                Mat& R,
                Mat& T){
    if(root!=string(rootpath)){

        root=string(rootpath);
        get_all(root, ".txt", txt);                            // gathers all filepaths with each suffix, into c++ vectors.
        get_all(root, ".png", png);
        get_all(root, ".depth", depth);
                cout<<"Loading......"<<endl;
    }
    std::string str = txt[imageNumber].c_str();
    char * ch = new char [str.length()+1];
    std::strcpy (ch, str.c_str());
    
	//const wchar_t* w = txt[imageNumber].c_str();
	//_bstr_t b(w);                                              // a MS comdef.h class. Basic string or binary string
	//const char* ch = b;
    convertAhandaPovRayToStandard(ch,R,T);
    //cout<<"Reading: "<<png[imageNumber].filename().string()<<endl;
    image = imread(png[imageNumber].string());
    int r=image.rows;
    int c=image.cols;
    if(depth.size()>0){
        cout<<"Depth: "<<depth[imageNumber].filename().string()<<endl;
        d=loadDepthAhanda(depth[imageNumber].string(), r,c,cameraMatrix);
    }
    
   
}


Mat loadDepthAhanda(string filename, int r,int c,Mat cameraMatrix){
    ifstream in(filename.c_str());
    int sz=r*c;
    Mat_<float> out(r,c);
    float * p=(float *)out.data;
    for(int i=0;i<sz;i++){
        in>>p[i];
        assert(p[i]!=0);
    }
    Mat_<double> K = cameraMatrix;
    double fx=K(0,0);
    double fy=K(1,1);
    double cx=K(0,2);
    double cy=K(1,2);
    for (int i=0;i<r;i++){
        for (int j=0;j<c;j++,p++){
            double x=j;
            double y=i;
            x=(x-cx)/fx;
            y=(y-cy)/fy;
            *p=*p/sqrt(x*x+y*y+1);
        }
    }
    
    
    
    return out;
}




#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 


// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
void get_all(const fs::path& root, const string& ext, vector<fs::path>& ret)
{  
  if (!fs::exists(root)) return;

  if (fs::is_directory(root))
  {
    typedef std::set<boost::filesystem::path> Files;
    Files files;
    fs::recursive_directory_iterator it0(root);
    fs::recursive_directory_iterator endit0;
    std::copy(it0, endit0, std::inserter(files, files.begin()));
    Files::iterator it= files.begin();
    Files::iterator endit= files.end();
    while(it != endit)
    {
      if (fs::is_regular_file(*it) && (*it).extension() == ext)
      {
        ret.push_back(*it);
      }
      ++it;
    }
  }
}
