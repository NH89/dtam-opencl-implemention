#pragma once
#include "utils.hpp"
#include "RunCL.h"
//typedef int     FrameID;
//typedef size_t  st;
//#define CONSTT uint  rows, uint  cols, uint  layers, uint layerStep, float* hdata, float* cdata, float* lo, float* hi, float* loInd, float3* base,  float* bf
//#define downp  (point+w)
//#define upp    (point-w)
//#define rightp (point+1)
//#define leftp  (point-1)
//#define here   (point)
//#define gdown   gd[here]
//#define gup     gu[here]
//#define gleft   gl[here]
//#define gright  gr[here]

class CostVol
{
public:
	~CostVol();
	CostVol(cv::Mat image, /*FrameID _fid,*/ // int _layers,  float _near,  float _far,
		cv::Mat R,  cv::Mat T,  cv::Mat _cameraMatrix,
		 //float occlusionThreshold,
		 //boost::filesystem::path out_path,
		 //float initialCost   = 1.0,//3.0,
		 //float initialWeight = .0000001,
		 Json::Value obj_//,
		 //int verbosity_ = -1
		 );
	Json::Value obj;
	RunCL   cvrc;
	//FrameID fid;
	int     rows, cols, layers, count, verbosity;
	//near & far : inverse depth of center of voxels in layer layers-1 //inverse depth of center of voxels in layer 0
	float   near, far,  depthStep;//, initialWeight, occlusionThreshold;
	Matx44f K, inv_K, pose, inv_pose;
	Mat R, T, cameraMatrix, projection;												//Note! should be in OpenCV format //projects world coordinates (x,y,z) into (rows,cols,layers)
	Mat baseImage, baseImageGray, costdata, hit, img_sum_data;
	Mat _qx, _qy, _d, _a, _g, _g1, _gx, _gy, lo, hi;

	void updateCost(const cv::Mat& image,  const cv::Mat& R,  const cv::Mat& T);	//Accepts pinned RGBA8888 or BGRA8888 for high speed
	void cacheGValues();
	void initializeAD();
	void initOptimization();
	void updateQD();
	bool updateA();
	void GetResult();
	void computeSigmas(float epsilon, float theta, float L);
	void writePointCloud(cv::Mat depthMap);

private:
	cv::Mat cBuffer;	//Must be pagable
	cv::Ptr<char> ref;
	void  solveProjection(const cv::Mat& R,  const cv::Mat& T);
	void  checkInputs(    const cv::Mat& R,  const cv::Mat& T,   const cv::Mat& _cameraMatrix);
	float old_theta, theta, thetaStart, thetaStep, thetaMin, epsilon, lambda, sigma_d, sigma_q;
};


