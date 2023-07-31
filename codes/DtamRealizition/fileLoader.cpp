#include <iostream>
#include <set>
#include <string>
#include "fileLoader.hpp"
#include <boost/filesystem.hpp>
#include <fstream>

using namespace cv;
using namespace std;
namespace fs = ::boost::filesystem;
static fs::path root;
static vector<fs::path> txt;
static vector<fs::path> png;
static vector<fs::path> depth;

void get_all(const fs::path& root, const string& ext, vector<fs::path>& ret);

void loadAhanda( std::string rootpath,
                 int imageNumber,
                 Mat& image,
                 Mat& d,
                 Mat& cameraMatrix,
                 Mat& R,
                 Mat& T,
                 int verbosity_){
    if(root!=rootpath){
        root=rootpath;
        get_all(root, ".txt", txt);                            // gathers all filepaths with each suffix, into c++ vectors.
        get_all(root, ".png", png);
        get_all(root, ".depth", depth);
                cout<<"Loading......"<<endl;
    }
    if(verbosity_>0) {
      cout << "\nfilepath: "<< string(rootpath) << "\n" << flush;
    }
    if(verbosity_>0) {  
      cout << "\n" << txt[imageNumber].c_str()  << flush;
      cout << "\n" << png[imageNumber].c_str()  << flush;
      cout << "\n" << depth[imageNumber].c_str();
      cout << "\n" << flush;
    }
    std::string str = txt[imageNumber].c_str();                // grab .txt filename from array (e.g. "scene_00_0000.txt")
    char        *ch = new char [str.length()+1];
    std::strcpy (ch, str.c_str());
    convertAhandaPovRayToStandard(ch,R,T,cameraMatrix);        // compute R, T & cameraMatrix from "*.txt"
    if(verbosity_>0) cout<<"Loading image......"<<endl<< flush;
    image = imread(png[imageNumber].string());                 // Read image
    /*
    cv::imshow("loadAhanda png", image);
    */
    if (image.type() == CV_32FC3) {}
    else if(image.type() == CV_8UC3){
      image.convertTo(image, CV_32F);                          // convert 8-bit uchar -> 32-bit float
      image /= 256;
    }else{
      std::cout << "\n\nFileloader.cpp, loadAhanda(..) :  unhandled image type "<< image.type() <<"\n"<<std::flush;
      exit(0);
    }
    /*
    Mat temp;
    image.convertTo(temp, CV_8U);                              // NB need CV_U8 for imshow(..)
    cv::imshow("loadAhanda CV_32F", temp);
    */
    cv::GaussianBlur(image, image, Size(3,3), 0, 0 );          // Gaussian blur to suppress image artefacts, that later affect photometric error. (9,9)
    int r = image.rows;
    int c = image.cols;
    if(depth.size()>0){
        if(verbosity_>0) cout<<"\nDepth: " <<depth[imageNumber].filename().string()<<"\t";
        d = loadDepthAhanda(depth[imageNumber].string(), r,c,cameraMatrix);
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
    Mat_<float> K = cameraMatrix;
    float fx=K(0,0);
    float fy=K(1,1);
    float cx=K(0,2);
    float cy=K(1,2);
    for (int i=0;i<r;i++){
        for (int j=0;j<c;j++,p++){
            float x=j;
            float y=i;
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
  if (!fs::exists(root))        return;
  if (fs::is_directory(root))   {
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
