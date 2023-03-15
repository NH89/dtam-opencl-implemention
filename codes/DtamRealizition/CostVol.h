#pragma once
#include "utils.hpp"
#include "RunCL.h"
typedef int     FrameID;
typedef size_t  st;
#define CONSTT uint  rows, uint  cols, uint  layers, uint layerStep, float* hdata, float* cdata, float* lo, float* hi, float* loInd, float3* base,  float* bf
#define downp  (point+w)
#define upp    (point-w)
#define rightp (point+1)
#define leftp  (point-1)
#define here   (point)
#define gdown   gd[here]
#define gup     gu[here]
#define gleft   gl[here]
#define gright  gr[here]




class CostVol
{
public:
	~CostVol();
	CostVol(cv::Mat image,  FrameID _fid,  int _layers,  float _near,  float _far,
		cv::Mat R,  cv::Mat T,  cv::Mat _cameraMatrix,
		 float occlusionThreshold,
		 boost::filesystem::path out_path,
		 float initialCost   = 1.0,//3.0,
		 float initialWeight = .001
		 );
	RunCL   cvrc;
	FrameID fid;
	int     rows, cols, layers, count;
	//near & far : inverse depth of center of voxels in layer layers-1 //inverse depth of center of voxels in layer 0
	float   near, far,  depthStep, initialWeight, occlusionThreshold;
	Matx44f K, inv_K, pose, inv_pose;
	Mat R, T, cameraMatrix, projection;					//Note! should be in OpenCV format
	Mat baseImage, baseImageGray, costdata, hit;  		//projects world coordinates (x,y,z) into (rows,cols,layers)
	Mat _qx, _qy, _d, _a, _g, _g1, _gx, _gy, lo, hi;

	void updateCost(const cv::Mat& image,  const cv::Mat& R,  const cv::Mat& T);		//Accepts pinned RGBA8888 or BGRA8888 for high speed
	void cacheGValues();
	void initializeAD();
	void initOptimization();
	void updateQD();
	bool updateA();
	void GetResult();

	void computeSigmas(float epsilon, float theta);/*{
		float lambda, alpha, gamma, delta, mu, rho, sigma;
		float L = 4;	//lower is better(longer steps), but in theory only >=4 is guaranteed to converge. For the adventurous, set to 2 or 1.44
		lambda  = 1.0 / theta;
		alpha   = epsilon;
		gamma   = lambda;
		delta   = alpha;
		mu      = 2.0*std::sqrt(gamma*delta) / L;
		rho     = mu / (2.0*gamma);
		sigma   = mu / (2.0*delta);
		sigma_d = rho;
		sigma_q = sigma;
	}*/

private:
	cv::Mat cBuffer;	//Must be pagable
	cv::Ptr<char> ref;
	void  solveProjection(const cv::Mat& R,  const cv::Mat& T);
	void  checkInputs(    const cv::Mat& R,  const cv::Mat& T,   const cv::Mat& _cameraMatrix);
	float old_theta, theta, thetaStart, thetaStep, thetaMin, epsilon, lambda, sigma_d, sigma_q;
};
