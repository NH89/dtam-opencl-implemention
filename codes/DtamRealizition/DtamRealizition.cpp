// DtamRealizition.cpp : 定义控制台应用程序的入口点。
//
//#include "stdafx.h"   // windows precompiled header file
//#include <boost/filesystem.hpp> // included from RunCl.h, via CostVol.h
#include "CostVol.h"

#include "fileLoader.hpp"
#include <iostream>		// /usr/include/c++/11/iostream
#include <iomanip>		// /usr/include/c++/11/iomanip
#include <cstdio>		// /usr/include/c++/11/cstdio
#include <ctime>		// /usr/include/c++/11/ctime : /usr/include/c++/11/tr1/ctime
#include <fstream>		// /usr/include/c++/11/fstream
#include <string>		// /usr/include/c++/11/string
#include <sstream>		// /usr/include/c++/11/sstream
//#include <filesystem>

using namespace cv;
using namespace std;
int main()
{
	cout << "\n main_chk 0\n" << flush;
	int numImg = 50;
	char filename[500];
	Mat image, R, T;						// TODO where do these values come from ? See "scene_00_*.txt flies and "getcamK_octave.m" in ICL dataset.
	Mat cameraMatrix = (Mat_<double>(3, 3) << 481.20,   0.0,  319.5,
												0.0,  480.0,  239.5,
												0.0,    0.0,    1.0);

	cout << "\n main_chk 1\n" << flush;  	// make output folders ////////////////////////////////////////////////////////////
	std::time_t   result  = std::time(nullptr);
	std::string   out_dir = std::asctime(std::localtime(&result));
	out_dir.pop_back(); 															// req to remove new_line from end of string.
	boost::filesystem::path out_path(boost::filesystem::current_path());
	out_path = out_path.parent_path();												// move "out_path" up two levels in the directory tree.
	out_path = out_path.parent_path();
	out_path += "/output/";// + out_dir;

	if(boost::filesystem::create_directory(out_path)) { std::cerr<< "Directory Created: "<<out_path<<std::endl;
	}else{ std::cerr<< "Directory previously created: "<<out_path<<std::endl;
	}
	out_path +=  out_dir;
	cout <<"Creating output directories: "<< out_path <<std::endl;
	boost::filesystem::create_directory(out_path);//, ec);
	out_path += "/";

	cout << "\n main_chk 1.2\n" << flush; ////////////////////////////////////////////////////////////////////////////////
	vector<Mat> images, Rs, ds, Ts, Rs0, Ts0, D0;
	double reconstructionScale = 1;
	int inc = 1;

	cout << "\n main_chk 2\n" << flush;
	for (int i =0; i < 11; i += inc) {												// Load images & data from file into c++ vectors
		Mat tmp, d, image;
		int offset = 462;//380;//0;// better location in this dataset for translation for paralax flow.
																					// TODO use a control file specifying where to sample the video.
		loadAhanda("/home/nick/programming/ComputerVision/DataSets/ahanda-icl/Trajectory_for_Variable_Frame-Rate/200fps/200fps_GT_archieve",
					//"/home/nick/programming/ComputerVision/DataSets/ahanda-icl/office_room/office_room_traj0_loop",
					//"/home/nick/programming/ComputerVision/DataSets/ahanda-icl/LivingRoom'lt kt0'/living_room_traj0_loop",
					//"/home/hockingsn/Programming/OpenCV/OpenDTAM/data/Trajectory_30_seconds",//"D:\\projects\\DTAM-master\\DTAM-master\\Trajectory_30_seconds\\",
			65535,
			i + offset,
			image,																	// NB .png image is converted CV_8UC3 -> CV_32FC3, and given GaussianBlurr (3,3).
			d,
			cameraMatrix,
			R,
			T);
		cout << ", image.size="<<image.size;

		images.push_back(image);
		Rs.push_back(R.clone());
		Ts.push_back(T.clone());
		ds.push_back(d.clone());
		Rs0.push_back(R.clone());
		Ts0.push_back(T.clone());
		D0.push_back(1 / d);
	}

	cout << "\n main_chk 3\n" << flush;
	//Setup camera matrix
	double sx 					= reconstructionScale;
	double sy 					= reconstructionScale;
	int layers 					= 32;
	int imagesPerCV 			= 10;
	float occlusionThreshold 	= .05;
	int startAt 				= 0;
	cout<<"images[startAt].size="<<images[startAt].size<<"\n";
                                                                                   // Instantiate CostVol ///////////
	CostVol cv(images[startAt], (FrameID)startAt, layers, 0.015, 0.0, Rs[startAt], Ts[startAt], cameraMatrix, occlusionThreshold, out_path);

//	return 0; //#################### early halt #################################

	cout << "\n main_chk 4\n" << flush;
	cout << "calculate cost volume: ================================================" << endl << flush;
	for (int imageNum = 1; imageNum < 10; imageNum++){                              // Update CostVol ////////////////
		cv.updateCost(images[imageNum], Rs[imageNum], Ts[imageNum]);
		cout<<"\ncv.updateCost: images["<<imageNum<<"].size="<<images[imageNum].size<<"\n";
	}

	cout << "\n main_chk 5\n" << flush;
	cout << "cacheGValues: =========================================================" << endl;
	cv.cacheGValues();                                                             // cacheGValues()  elementwise weighting on keframe image gradient

//	cv.initializeAD();

	cout << "\n main_chk 6\n" << flush;
	cout << "optimizing: ===========================================================" << endl;
	bool doneOptimizing;
	do {
		for (int i = 0; i < 10; i++)
			cv.updateQD();                                                         // Optimize Q, D   (primal-dual)
		doneOptimizing = cv.updateA();                                             // Optimize A      (pointwise exhaustive search)
		doneOptimizing = true; // debug single loop TODO remove this line.
	} while (!doneOptimizing);

	cout << "\n main_chk 7\n" << flush;
	cv::Mat depthMap;
	double min = 0, max = 0;
	cv::Mat depthImg;

	cout << "\n main_chk 8\n" << flush;
	cv.GetResult();                                                                // GetResult /////////////////////
    depthMap = cv._a;

	double minVal=1, maxVal=1;
	cv::Point minLoc={0,0}, maxLoc{0,0};
	cv::minMaxLoc(depthMap, &minVal, &maxVal, &minLoc, &maxLoc);

	std::stringstream ss;
	ss << "Depthmap, maxVal: " << maxVal << " minVal: " << minVal << " minLoc: "<< minLoc <<" maxLoc: " << maxLoc;
	cv::imshow(ss.str(), (depthMap*(1.0/maxVal)));
	std::cout << ss.str() << std::endl<<flush;
	cv::waitKey(-1);

	cout << "\n main_chk 9 : finished\n" << flush;
	//return 0;
	exit(0);
}

