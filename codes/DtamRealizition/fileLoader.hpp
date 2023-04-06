#ifndef FILE_LOADER
#define FILE_LOADER
#include <opencv2/opencv.hpp> //"/usr/local/include/opencv4/opencv2/opencv.hpp"//"opencv\cv.h"
// loads all files of a given name and extension
#include "convertAhandaPovRayToStandard.h"
void loadAhanda(std::string rootpath, //const char * rootpath,
                //float range,
                int imageNumber,
                cv::Mat& image,
                cv::Mat& d,
                cv::Mat& cameraMatrix,
                cv::Mat& R,
                cv::Mat& T,
                int verbosity_ = -1);
cv::Mat loadDepthAhanda(std::string filename, int r,int c,cv::Mat cameraMatrix);

#endif
