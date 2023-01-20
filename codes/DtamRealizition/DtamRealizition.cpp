// DtamRealizition.cpp : 定义控制台应用程序的入口点。
//
//#include "stdafx.h"   // windows precompiled header file
#include <boost/filesystem.hpp>
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

	//initialize opencl
	int numImg = 50;
	char filename[500];
	//std::string pathDepthDir = "D:\\projects\\";
	Mat image, R, T;
	Mat cameraMatrix = (Mat_<double>(3, 3) << 481.20, 0.0, 319.5,  // TODO where do these values come from ? See "scene_00_*.txt flies and "getcamK_octave.m" in ICL dataset.
		0.0, 480.0, 239.5,
		0.0, 0.0, 1.0);
	cout << "\n main_chk 1\n" << flush;

	// make output directory
	std::time_t result = std::time(nullptr);
	const std::string out_dir = std::asctime(std::localtime(&result));
	boost::filesystem::path out_path(boost::filesystem::current_path());
	out_path += "/../../output/";// + out_dir;

	if(boost::filesystem::create_directory(out_path)) {
		std::cerr<< "Directory Created: "<<out_path<<std::endl;
	}else{ std::cerr<< "Directory previously created: "<<out_path<<std::endl;
	}
	out_path +=  out_dir;
	cout <<"Creating output directories: "<< out_path <<std::endl;
	boost::filesystem::create_directory(out_path);//, ec);
	boost::filesystem::path temp_path = out_path;
	temp_path += "/a";
	boost::filesystem::create_directory(temp_path);
	temp_path = out_path;
	temp_path += "/b";
	boost::filesystem::create_directory(temp_path);

	cout << "\n main_chk 1.2\n" << flush;
	///////
	return 0;

	vector<Mat> images, Rs, ds, Ts, Rs0, Ts0, D0;
	double reconstructionScale = 1;
	int inc = 1;

	cout << "\n main_chk 2\n" << flush;
	for (int i =0; i < 11; i += inc) {                                             // Load images & data from file into c++ vectors
		Mat tmp, d, image;
		int offset = 462;//0;// better location in this dataset for translation for paralax flow.#################### TODO use a control file specifying where to sample the video.
		//cout << "chk 2.1\n" << flush;
		//cout << "i = " << i << ", offset = " << offset << ".\n" << flush;

		loadAhanda("/home/nick/programming/ComputerVision/DataSets/ahanda-icl/office_room/office_room_traj0_loop", // NB need to replace with launch file
                   //"/home/hockingsn/Programming/OpenCV/OpenDTAM/data/Trajectory_30_seconds",//"D:\\projects\\DTAM-master\\DTAM-master\\Trajectory_30_seconds\\",
			65535,
			i + offset,
			image,
			d,
			cameraMatrix,
			R,
			T);
		//cout << "chk 2.2\n" << flush;
		cout << ", image.size="<<image.size;

		images.push_back(image);
		Rs.push_back(R.clone());
		Ts.push_back(T.clone());
		ds.push_back(d.clone());
		Rs0.push_back(R.clone());
		Ts0.push_back(T.clone());
		D0.push_back(1 / d);
		//cout << "chk 2.3\n" << flush;
	}

	cout << "\n main_chk 3\n" << flush;
	//Setup camera matrix
	double sx = reconstructionScale;
	double sy = reconstructionScale;

	int layers = 32;
	int imagesPerCV = 10;
	float occlusionThreshold = .05;

	int startAt = 0;

	cout<<"images[startAt].size="<<images[startAt].size<<"\n";
                                                                                   // Instantiate CostVol ///////////
	CostVol cv(images[startAt], (FrameID)startAt, layers, 0.015, 0.0, Rs[startAt], Ts[startAt], cameraMatrix, occlusionThreshold);


	cout << "\n main_chk 4\n" << flush;
	cout << "calculate cost volume: ================================================" << endl << flush;

	for (int imageNum = 1; imageNum < 10; imageNum++){                              // Update CostVol ////////////////
		cv.updateCost(images[imageNum], Rs[imageNum], Ts[imageNum]);
		cout<<"\ncv.updateCost: images["<<imageNum<<"].size="<<images[imageNum].size<<"\n";
	}

	cout << "\n main_chk 5\n" << flush;
	cout << "cacheGValues: =========================================================" << endl;
	cv.cacheGValues();                                                             // cacheGValues()  elementwise weighting on keframe image gradient

	cout << "\n main_chk 6\n" << flush;
	cout << "optimizing: ===========================================================" << endl;
	bool doneOptimizing;

	do
	{
		for (int i = 0; i < 10; i++)
			cv.updateQD();                                                         // Optimize Q, D   (primal-dual)
		doneOptimizing = cv.updateA();                                             // Optimize A      (pointwise exhaustive search)
	} while (!doneOptimizing);

	cout << "\n main_chk 7\n" << flush;
	cout << "done: " << endl;
	
	cv::Mat depthMap;
	double min = 0, max = 0;
	cv::Mat depthImg;
	cv::namedWindow("depthmap", 1);

	cout << "\n main_chk 8\n" << flush;

	cv.GetResult();                                                                // GetResult /////////////////////
    depthMap = cv._a;                                                              // depthMap = cv._a

    cout << "\n main_chk 9\n" << flush;

	cv::minMaxIdx(depthMap, &min, &max);
	std::cout << "Max: " << max << " Min: " << min << std::endl<<flush;
	depthMap.convertTo(depthImg, CV_8U, 255 / max, 0);
	medianBlur(depthImg, depthImg, 3);

	cout << "\n main_chk 10\n" << flush;
	cv::imshow("depthmap", depthImg);                                             //cv::imshow("depthmap", depthImg);
	cv::waitKey(-1);

	cout << "\n main_chk 11\n" << flush;
	return 0;
}

