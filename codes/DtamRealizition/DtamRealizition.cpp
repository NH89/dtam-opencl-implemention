
#include "CostVol.h"
#include "fileLoader.hpp"
#include <iostream>		// /usr/include/c++/11/iostream
#include <iomanip>		// /usr/include/c++/11/iomanip
#include <cstdio>		// /usr/include/c++/11/cstdio
#include <ctime>		// /usr/include/c++/11/ctime : /usr/include/c++/11/tr1/ctime
#include <fstream>		// /usr/include/c++/11/fstream
#include <string>		// /usr/include/c++/11/string
#include <sstream>		// /usr/include/c++/11/sstream
#include <jsoncpp/json/json.h> // TODO add to CMakeLiss.txt as findpackege(jsoncpp required)

#define DATASETS "/home/hockingsn/Programming/computer_vision_data/"   //  /home/nick/programming/ComputerVision/DataSets

using namespace cv;
using namespace std;
int main(int argc, char *argv[])
{
	if (argc !=2) { cout << "\n\nUsage : DTAM_OpenCL <config_file.json>\n\n" << flush; exit; }
	ifstream ifs(argv[1]);
    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj);
	
	int verbosity_ 		= obj["verbosity"].asInt() ;						// -1= none, 0=errors only, 1=basic, 2=lots.
	int imagesPerCV 	= obj["imagesPerCV"].asUInt() ; 					//30;//7; //30;
	if(verbosity_>0) cout << "\n main_chk 0\n" << flush;
	
	Mat image, R, T;														// See "scene_00_*.txt flies and "getcamK_octave.m" in ICL dataset.
	float cam_matrix[9];
	for (int i=0; i<9; i++){cam_matrix[i] = obj["cameraMatrix"][i].asFloat(); }
	Mat cameraMatrix 	= Mat(Size(3,3), CV_32FC1 , cam_matrix  );			// TODO use Matx instead ?

	if(verbosity_>0) cout << "\n main_chk 1\n" << flush;  					// make output folders //
	std::time_t   result  = std::time(nullptr);
	std::string   out_dir = std::asctime(std::localtime(&result));
	out_dir.pop_back(); 													// req to remove new_line from end of string.
	boost::filesystem::path out_path(boost::filesystem::current_path());
	out_path = out_path.parent_path().parent_path();						// move "out_path" up two levels in the directory tree.
	out_path += "/output/";

	if(boost::filesystem::create_directory(out_path)) { std::cerr<< "Directory Created: "<<out_path<<std::endl;
	}else{ std::cerr<< "Directory previously created: "<<out_path<<std::endl;
	}
	out_path +=  out_dir;
	if(verbosity_>0) cout <<"Creating output directories: "<< out_path <<std::endl;
	boost::filesystem::create_directory(out_path);
	out_path += "/";

	if(verbosity_>0) cout << "\n main_chk 1.2\n" << flush;
	vector<Mat> images, Rs, ds, Ts, Rs0, Ts0, D0;
	float reconstructionScale = 1;
	int inc = 1;

	if(verbosity_>0) cout << "\n main_chk 2\n" << flush;
	stringstream ss0;
	ss0 << DATASETS << "ahanda-icl/Trajectory_for_Variable_Frame-Rate/200fps/200fps_GT_archieve";
	
	for (int i =0; i < imagesPerCV; i += inc) {								// Load images & data from file into c++ vectors
		Mat tmp, d, image;
		int offset = 0;					//475;//10;//46;//462;//380;//0;	// better location in this dataset for translation for paralax flow.
																			// TODO use a control file specifying where to sample the video.
		loadAhanda( ss0.str(),
					i + offset,
					image,													// NB .png image is converted CV_8UC3 -> CV_32FC3, and given GaussianBlurr (3,3).
					d,
					cameraMatrix,											// NB recomputes cameraMatrix for each image, but hard codes 640x480.
					R,
					T);
		if(verbosity_>0) cout << ", image.size="<<image.size << "\ncameraMatrix.rows="<<cameraMatrix.rows<<",  cameraMatrix.cols="<<cameraMatrix.rows<<"\n"<<flush;

		images.push_back(image);
		Rs.push_back(R.clone());
		Ts.push_back(T.clone());
		ds.push_back(d.clone());
		Rs0.push_back(R.clone());
		Ts0.push_back(T.clone());
		D0.push_back(1 / d);
	}

	if(verbosity_>0){
		cout<<"\ncameraMatrix.type()="<<cameraMatrix.type()<< "\ntype 5 = CV_32FC1\n"<<std::flush;
		int rows = cameraMatrix.rows;
		int cols = cameraMatrix.cols;
		
		for (int i=0; i<rows; i++){
			for (int j=0; j<cols; j++){
				cout<<", "<<cameraMatrix.at<float>(i,j)<< std::flush;
			}cout<<"\n"<< std::flush;
		}cout<<"\n"<< std::flush;
		
		cout << "\n main_chk 3\n" << flush; 									//Setup camera matrix
	}
	float sx 					= reconstructionScale;
	float sy 					= reconstructionScale;
	int   layers 				= 256;//128;//256; //64;										//32;//
	float min_inv_depth 		= 1.0/450.0;	//1.0/4347.0; //							//far// NB see values in DTAM_mapping/input/*.json file for each dataset.
	float max_inv_depth 		= 1.0/70.0;	//1.0/3289.0; //
	float occlusionThreshold 	= .05;
	int   startAt 				= 0;
	if(verbosity_>0) cout<<"images[startAt].size="<<images[startAt].size<<"\n";

	if(verbosity_>0) cout << "\n main_chk 3.1\n" << flush; 									// Instantiate CostVol ///////////
	CostVol cv(images[startAt], (FrameID)startAt, layers, max_inv_depth, min_inv_depth, Rs[startAt], Ts[startAt], cameraMatrix, occlusionThreshold, out_path /* float initialCost=1.0, float initialWeight=0.001*/ );

	if(verbosity_>0) cout << "\n main_chk 4\tcalculate cost volume: ================================================" << endl << flush;
	for (int imageNum = 1; imageNum < imagesPerCV; imageNum+=1){			// Update CostVol ////////////////
		cv.updateCost(images[imageNum], Rs[imageNum], Ts[imageNum]);
		if(verbosity_>0) {
			cout<<"\ncv.updateCost: images["<<imageNum<<"].size="<<images[imageNum].size<<"\n";
			if (imageNum%5 == 1) cv.cvrc.saveCostVols(imageNum+1);
		}
	}

	if(verbosity_>0) cout << "\n main_chk 5\tcacheGValues: =========================================================" << endl<<flush;
	cv.cacheGValues();														// cacheGValues()  elementwise weighting on keframe image gradient

	if(verbosity_>0) cout << "\n main_chk 6\toptimizing: ===========================================================" << endl<<flush;
	bool doneOptimizing;
	int opt_count = 0;
	do {
		for (int i = 0; i < 10; i++)
			cv.updateQD();													// Optimize Q, D   (primal-dual)
		doneOptimizing = cv.updateA();										// Optimize A      (pointwise exhaustive search)
		opt_count ++;
	} while (!doneOptimizing && (opt_count<60));  //60

	if(verbosity_>0) cout << "\n main_chk 7\n" << flush;
	cv::Mat depthMap;
	cv.GetResult();															// GetResult /////////////////////
    depthMap 		= cv._a;
	double minVal	=1, 	maxVal=1;
	cv::Point minLoc={0,0}, maxLoc{0,0};
	cv::minMaxLoc(depthMap, &minVal, &maxVal, &minLoc, &maxLoc);
	std::stringstream ss;
	ss << "Depthmap, maxVal: " << maxVal << " minVal: " << minVal << " minLoc: "<< minLoc <<" maxLoc: " << maxLoc;
	cv::imshow(ss.str(), (depthMap*(1.0/maxVal)));
	if(verbosity_>0) std::cout << ss.str() << std::endl<<flush;
	cv::waitKey(-1);

	cout << "\n main_chk 9 : finished\n" << flush;
	exit(0);
}
