#include "CostVol.h"
#include <fstream>
#include <iomanip>   // std::setprecision, std::setw

using namespace cv;
using namespace std;

#define FLATALLOC(n) n.create(rows, cols, CV_32FC1); n.reshape(0, rows);

CostVol::~CostVol()
{
}

void CostVol::solveProjection(const cv::Mat& R, const cv::Mat& T) {
	Mat P;
	RTToP(R, T, P);
																			//P:4x4 rigid 3D transformation taking points from 3D world to the 3D camera frame
																			//r1 r2 r3 tx
																			//r4 r5 r6 ty
																			//r7 r8 r9 tz
																			//0  0  0  0

																			//Camera:
																			//fx 0  cx
																			//0  fy cy
																			//0  0  1
	projection.create(4, 4, CV_32FC1);										// NB cv::Mat projection; declared in CostVol.h
	projection = 0.0;
	projection(Range(0, 2), Range(0, 3)) += cameraMatrix.rowRange(0, 2);
																			//Projection:
																			//fx 0  cx 0
																			//0  fy cy 0
																			//0  0  0  0
																			//0  0  0  0
	projection.at<float>(2, 3) = 1.0;
	projection.at<float>(3, 2) = 1.0;
																			//Projection: Takes camera coordinates to pixel coordinates:x_px,y_px,1/zc
																			//fx 0  cx 0
																			//0  fy cy 0
																			//0  0  0  1
																			//0  0  1  0
	Mat originShift = (Mat)(Mat_<float>(4, 4) << 	1.0, 0.,   0.,   0.,
													0.,  1.0,  0.,   0.,
													0.,  0.,   1.0,  -far,
													0.,  0.,   0.,   1.0);
	projection = originShift*projection;									//put the origin at 1/z_ from_camera_center = far.   NB main() line 113, costvol constructor, far=0.0 .
	projection.row(2) /= depthStep;											//stretch inverse depth so now   x_cam,y_cam,z_cam-->x_cv_px, y_cv_px , [1/z_from_camera_center - far]_px
	projection = projection*P;												//projection now goes     x_world,y_world,z_world -->x_cv_px, y_cv_px , [1/z_from_camera_center - far]_px
}

void CostVol::checkInputs(
    const cv::Mat& R, 
    const cv::Mat& T,
	const cv::Mat& _cameraMatrix)
{
																			if(verbosity>0)  cout<<"\ncheckInputs\n"<< std::flush;
	assert(R.size() == Size(3, 3));
	assert(R.type() == CV_32FC1);
	assert(T.size() == Size(1, 3));
	assert(T.type() == CV_32FC1);
	assert(_cameraMatrix.size() == Size(3, 3));
	assert(_cameraMatrix.type() == CV_32FC1);
																			if(verbosity>0) {	// print cameraMatrix
																				cout<<"\ncameraMatrix.rows="<<cameraMatrix.rows<<",  cameraMatrix.cols="<<cameraMatrix.cols<<"\n"<< std::flush;   // empty Mat why ?
																				for (int i=0; i<cameraMatrix.rows; i++){
																					for (int j=0; j<cameraMatrix.cols; j++){
																						std::cout<<", "<<cameraMatrix.at<float>(i,j)<< std::flush;
																					}std::cout<<"\n"<< std::flush;
																				}std::cout<<"\n"<< std::flush;
																			}
	CV_Assert(_cameraMatrix.at<float>(2, 0) == 0.0);
	CV_Assert(_cameraMatrix.at<float>(2, 1) == 0.0);
	CV_Assert(_cameraMatrix.at<float>(2, 2) == 1.0);
}

CostVol::CostVol(
	Mat 		image,
	//FrameID 	_fid,
	//int 		_layers,
	//float 		_near,
	//float 		_far,
	cv::Mat 	R,
	cv::Mat 	T,
	cv::Mat 	_cameraMatrix,
	//float 		occlusionThreshold,
	boost::filesystem::path 	out_path,
	//float 		initialCost,	 															// NB defaults: float initialCost = 3.0, float initialWeight = .001
	//float 		initialWeight,
	Json::Value obj,
	int			verbosity_		// obj[""].asFloat()
) : cvrc(out_path, verbosity_), initialWeight(obj["initialWeight"].asFloat()), occlusionThreshold(obj["occlusionThreshold"].asFloat()), R(R), T(T)  // constructors for member classes //
{
	verbosity = verbosity_;
																							if(verbosity>0) cout << "CostVol_chk 0\n" << flush;
																							/*
																							*  Inverse of a transformation matrix:
																							* http://www.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche0053
																							*
																							*   {     |   }-1       {       |        }
																							*   {  R  | t }     =   {  R^T  |-R^T .t }
																							*   {_____|___}         {_______|________}
																							*   {0 0 0| 1 }         {0  0  0|    1   }
																							*
																							*/
	pose = pose.zeros();																	// pose
																							if(verbosity>0) cout << "CostVol_chk 1\n" << flush;
	for (int i=0; i<9; i++) pose.operator()(i/3,i%3) = R.at<float>(i/3,i%3);
	for (int i=0; i<3; i++) pose.operator()(i,3)     = T.at<float>(i);
	pose.operator()(3,3) = 1;																if(verbosity>0) cout << "CostVol_chk 2\n" << flush;
																							// inv_pose
	for (int i=0; i<3; i++) { for (int j=0; j<3; j++)  { inv_pose.operator()(i,j) = pose.operator()(j,i); } }
	cv::Mat inv_T = -R.t()*T;
	for (int i=0; i<3; i++) inv_pose.operator()(i,3) = inv_T.at<float>(i);
	for (int i=0; i<4; i++) inv_pose.operator()(3,i) = pose.operator()(3,i);

	K = K.zeros();																			// NB currently "cameraMatrix" found by convertAhandPovRay, called by fileLoader
	for (int i=0; i<9; i++) K.operator()(i/3,i%3) = _cameraMatrix.at<float>(i/3,i%3);		// TODO later, restructure mainloop to build costvol continuously...
	K.operator()(3,3)  = 1;
																							if(verbosity>1) {
																								cv::Matx44f test_pose = pose * inv_pose;
																								cout<<"\n\ninv_pose\n";									// Verify inv_pose:////////////////////////////////////////////////////
																								for(int i=0; i<4; i++){
																									for(int j=0; j<4; j++){
																										cout<<"\t"<< std::setw(5)<<inv_pose.operator()(i,j);
																									}cout<<"\n";
																								}cout<<"\n";

																								cout<<"\n\ntest_pose inversion: pose * inv_pose;\n";
																								for(int i=0; i<4; i++){
																									for(int j=0; j<4; j++){
																										cout<<"\t"<< std::setw(5)<<test_pose.operator()(i,j);
																									}cout<<"\n";
																								}cout<<"\n";

																								cout<<"\n\ntest_pose inversion: inv_pose * pose;\n";
																								test_pose = inv_pose * pose;
																								for(int i=0; i<4; i++){
																									for(int j=0; j<4; j++){
																										cout<<"\t"<< std::setw(5)<<test_pose.operator()(i,j);
																									}cout<<"\n";
																								}cout<<"\n";

																								for (int i=0; i<9; i++) {
																									cout<<"\ni="<<i<<","<<flush;
																									cout<<"\tK.operator()(i/3,i%3)="<<K.operator()(i/3,i%3)<<","<<flush;
																									cout<<"\tcameraMatrix.at<float>(i/3,i%3)="<<_cameraMatrix.at<float>(i/3,i%3)<<","<<flush;
																								}
																							}
																							if(verbosity>0) cout << "\n\nCostVol_chk 4\n" << flush;
	float fx   =  K.operator()(0,0);
	float fy   =  K.operator()(1,1);
	float skew =  K.operator()(0,1);
	float cx   =  K.operator()(0,2);
	float cy   =  K.operator()(1,2);														if(verbosity>0) {
																								cout<<"\nfx="<<fx <<"\nfy="<<fy <<"\nskew="<<skew <<"\ncx="<<cx <<"\ncy= "<<cy;
																								cout << "\n\nCostVol_chk 5\n" << flush;
																							}
	///////////////////////////////////////////////////////////////////// Inverse camera intrinsic matrix, see:
	// https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/#:~:text=Inverse,lines%20from%20the%20camera%20center.
	//float scalar = 1/(fx*fy);
	inv_K = inv_K.zeros();
	inv_K.operator()(0,0)  = 1.0/fx;  cout<<"\n1.0/fx="<<1.0/fx;
	inv_K.operator()(1,1)  = 1.0/fy;  cout<<"\n1.0/fy="<<1.0/fy;
	inv_K.operator()(2,2)  = 1.0;
	inv_K.operator()(3,3)  = 1.0;															// 1.0/(fx*fy);	// changed from (3,3) to correct from orthographic to perspective... TODO chk this works!

	inv_K.operator()(0,1)  = -skew/(fx*fy);
	inv_K.operator()(0,2)  = (cy*skew - cx*fy)/(fx*fy);
	inv_K.operator()(1,2)  = -cy/fy;
																							if(verbosity>-1) {
																								cv::Matx44f test_K = inv_K * K;
																								cout<<"\n\ntest_camera_intrinsic_matrix inversion\n";	// Verify inv_K:
																								for(int i=0; i<4; i++){
																									for(int j=0; j<4; j++){
																										cout<<"\t"<< std::setw(5)<<test_K.operator()(i,j);
																									}cout<<"\n";
																								}cout<<"\n";

																								std::cout << std::fixed << std::setprecision(-1);		// Inspect values in matricies ///////
																								cout<<"\n\npose\n";
																								for(int i=0; i<4; i++){
																									for(int j=0; j<4; j++){
																										cout<<"\t"<< std::setw(5)<<pose.operator()(i,j);
																									}cout<<"\n";
																								}cout<<"\n";

																								cout<<"\n\ninv_pose\n";
																								for(int i=0; i<4; i++){
																									for(int j=0; j<4; j++){
																										cout<<"\t"<< std::setw(5)<<inv_pose.operator()(i,j);
																									}cout<<"\n";
																								}cout<<"\n";

																								cout<<"\n\nK\n";
																								for(int i=0; i<4; i++){
																									for(int j=0; j<4; j++){
																										cout<<"\t"<< std::setw(5)<<K.operator()(i,j);
																									}cout<<"\n";
																								}cout<<"\n";

																								cout<<"\n\ninv_K\n";
																								for(int i=0; i<4; i++){
																									for(int j=0; j<4; j++){
																										cout<<"\t"<< std::setw(5)<<inv_K.operator()(i,j);
																									}cout<<"\n";
																								}cout<<"\n";
																							}
																							// For performance reasons, OpenDTAM only supports multiple of 32 image sizes with cols >= 64
	CV_Assert(image.rows % 32 == 0 && image.cols % 32 == 0 && image.cols >= 64);
																							if(verbosity>0) cout << "CostVol_chk 1\n" << flush;
	checkInputs(R, T, _cameraMatrix);
	//fid			= _fid;
	rows		= image.rows;
	cols		= image.cols;
	layers		= obj["layers"].asFloat();
	near		= obj["min_depth"].asFloat();
	far			= obj["max_depth"].asFloat();

	cvrc.params[PIXELS] 		= rows*cols;
	cvrc.params[ROWS] 			= rows;
	cvrc.params[COLS] 			= cols;
	cvrc.params[LAYERS] 		= layers;
	cvrc.params[MAX_INV_DEPTH] 	= 1.0/near;
	cvrc.params[MIN_INV_DEPTH] 	= 1.0/far;
	cvrc.params[INV_DEPTH_STEP] = (near - far) / (layers - 1);

	cameraMatrix = _cameraMatrix.clone();
	solveProjection(R, T);																	// solve projection
																							if(verbosity>0) {
																								cout << "CostVol_chk 1.2 \n";
																								cout<< "layers="<<cvrc.params[LAYERS]<<",\tcvrc.params[INV_DEPTH_STEP]="<< cvrc.params[INV_DEPTH_STEP];
																								cout<<"=("<<cvrc.params[MAX_INV_DEPTH]<<" - "<< cvrc.params[MIN_INV_DEPTH] <<") / ("<< cvrc.params[LAYERS] <<" -1 )\n" << flush;
																							}
	costdata = Mat::ones(layers, rows * cols, CV_32FC1);
	costdata = obj["initialCost"].asFloat();
	hit      = Mat::zeros(layers, rows * cols, CV_32FC1);
	hit      = obj["initialWeight"].asFloat();
	img_sum_data = Mat::zeros(layers, rows * cols, CV_32FC1);
	// #define FLATALLOC(n) n.create(rows, cols, CV_32FC1); n.reshape(0, rows);
	//FLATALLOC(lo);																			// TODO get rid of this. Get to
	//FLATALLOC(hi);																			// (i)  one record of rows & cols,
	FLATALLOC(_a);	// used for CostVol::GetResult(..)										// (ii) one allocation of buffers for data. -> DownLoadAndSave(..) offload & display use temp Mat objects.
	//FLATALLOC(_d);
	FLATALLOC(_gx); // used for line 277 cvrc.allocatemem(...)
	FLATALLOC(_gy);
    //FLATALLOC(_g1); // not used 
	_gx = _gy   = 1;
	cvrc.width  = cols;
	cvrc.height = rows;

	image.copyTo(baseImage);
	cvtColor(baseImage, baseImageGray, CV_RGB2GRAY);
	baseImageGray 	= baseImageGray.reshape(0, rows);										// baseImageGray used by CostVol::cacheGValues( cvrc.cacheGValue (baseImageGray));

	float *to_ptr=(float*)img_sum_data.data,  *from_ptr=(float*)baseImageGray.data;
	for (int i=0; i<img_sum_data.total(); i++) to_ptr[i]=from_ptr[i%baseImageGray.total()];	// tile baseImageGray onto img_sum_data.

	count      = 0;
	thetaStart = 0.2;				//0.2;		//200.0*off;								// DTAM paper & DTAM_Mapping has Theta_start = 0.2
	//thetaMin   = 9.9e-8;			//1.0e-4;	//1.0*off;									// DTAM paper NB theta depends on (i)photometric scale, (ii)num layers, (iii) inv_depth scale
	thetaStep  = 0.87;				//0.97;
	epsilon    = 1.0e-4;			//.1*off;
	lambda     = 1.0;				//1.0;		//.001 / off;								// DTAM paper: lambda = 1 for 1st key frame, and 1/(1+0.5*min_depth) thereafter
	theta      = thetaStart;
	old_theta  = theta;

	computeSigmas(epsilon, theta);																// lambda, theta, sigma_d, sigma_q set by CostVol::computeSigmas(..) in CostVol.h, called in CostVol::updateQD() below.
	cvrc.params[BETA_G]			=  obj["beta_g"].asFloat();										//1.5;														// DTAM paper : beta=0.001 while theta>0.001, else beta=0.0001
	cvrc.params[ALPHA_G]		=  obj["alpha_g"].asFloat()* pow(256,cvrc.params[BETA_G]);		//0.015 * pow(256,cvrc.params[BETA_G]);					///  __kernel void CacheG4, with correction for CV_8UC3 -> CV_32FC3
	cvrc.params[EPSILON]		=  obj["epsilon"].asFloat();									//0.1;			///  __kernel void UpdateQD					// epsilon = 0.1
	cvrc.params[SIGMA_Q]		=  0.0559017;													// sigma_q = 0.0559017
	cvrc.params[SIGMA_D]		=  sigma_d;
	cvrc.params[THETA]			=  theta;
	cvrc.params[LAMBDA]			=  obj["lambda"].asFloat();										//lambda;		///   __kernel void UpdateA2
	cvrc.params[SCALE_EAUX]		=  obj["scale_E_aux"].asFloat();								//10000;		// from DTAM_Mapping input/json/icl_numin.json    //1.0;

	cvrc.allocatemem( (float*)_gx.data, (float*)_gy.data, cvrc.params, layers, baseImage, (float*)costdata.data, (float*)hit.data, (float*)img_sum_data.data );
																							if(verbosity>0) cout << "CostVol_chk 6\n" << flush;
}

void CostVol::computeSigmas(float epsilon, float theta){
		float lambda, alpha, gamma, delta, mu, rho, sigma;
		float L = 4;							//lower is better(longer steps), but in theory only >=4 is guaranteed to converge. For the adventurous, set to 2 or 1.44
		lambda  = 1.0 / theta;
		alpha   = epsilon;
		gamma   = lambda;
		delta   = alpha;
		mu      = 2.0*std::sqrt(gamma*delta) / L;
		rho     = mu / (2.0*gamma);
		sigma   = mu / (2.0*delta);
		sigma_d = rho;
		sigma_q = sigma;
}

void CostVol::updateCost(const Mat& _image, const cv::Mat& R, const cv::Mat& T) 
{
	if(verbosity>0) cout << "\nupdateCost chk0," << flush;			// assemble 4x4 cam2cam homogeneous reprojection matrix:   cam2cam = K(4x4) * RT(4x4) * k^-1(4x4)
	Mat image;														// NB If K changes, then : K from current frame (to be sampled), and K^-1 from keyframe.
	_image.copyTo(image);
	cv::Matx44f K = cv::Matx44f::zeros();												// NB currently "cameraMatrix" found by convertAhandPovRay, called by fileLoader
	for (int i=0; i<9; i++) K.operator()(i/3,i%3) = cameraMatrix.at<float>(i/3,i%3);	// TODO later, restructure mainloop to build costvol continuously...
	K.operator()(3,3)  = 1;

	if(verbosity>1) {
		cout << "\nupdateCost chk1," << flush;
		cout << "\ncameraMatrix.size="<<cameraMatrix.size<<"\n"<<flush;   				// R.size=3 x 3,	T.size=3 x 1
		cout<<"\n\n'K' camera intrinsic matrix\n";
		for(int i=0; i<4; i++){
			for(int j=0; j<4; j++){
				cout<<"\t"<< std::setw(5)<<K.operator()(i,j);
			}cout<<"\n";
		}cout<<"\n";
	}
	std::cout << std::fixed << std::setprecision(-1);
	//https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/#:~:text=Inverse,lines%20from%20the%20camera%20center.
	float fx   =  K.operator()(0,0);
	float fy   =  K.operator()(1,1);
	float skew =  K.operator()(0,1);
	float cx   =  K.operator()(0,2);
	float cy   =  K.operator()(1,2);
	if(verbosity>1) {
		cout<<"\nfx="  <<fx;
		cout<<"\nfy="  <<fy;
		cout<<"\nskew="<<skew;
		cout<<"\ncx="  <<cx;
		cout<<"\ncy="  <<cy;
		cout<<"\nR.size="<<R.size<<",\tT.size="<<T.size<<"\n"<<flush;   				// R.size=3 x 3,	T.size=3 x 1
	}
	cv::Matx44f poseTransform = cv::Matx44f::zeros();
	for (int i=0; i<9; i++) poseTransform.operator()(i/3,i%3) = R.at<float>(i/3,i%3);
	for (int i=0; i<3; i++) poseTransform.operator()(i,3)     = T.at<float>(i);			// why is T so large ?
	poseTransform.operator()(3,3) = 1;

	cv::Matx44f cam2cam = K * poseTransform * inv_pose *  inv_K;						// cam2cam pixel transform, NB requires Pixel=(u,v,1,1/z)^T
	if(verbosity>1) {
		std::cout << std::fixed << std::setprecision(2);								//  Inspect values in matricies ///////
		cout<<"\n\nposeTransform\n";
		for(int i=0; i<4; i++){
			for(int j=0; j<4; j++){
				cout<<"\t"<< std::setw(5)<<poseTransform.operator()(i,j);
			}cout<<"\n";
		}cout<<"\n";

		cout<<"\n\nkeyframe_pose\n";
		for(int i=0; i<4; i++){
			for(int j=0; j<4; j++){
				cout<<"\t"<< std::setw(5)<<pose.operator()(i,j);
			}cout<<"\n";
		}cout<<"\n";

		cout<<"\n\ninv_pose\n";
		for(int i=0; i<4; i++){
			for(int j=0; j<4; j++){
				cout<<"\t"<< std::setw(5)<<inv_pose.operator()(i,j);
			}cout<<"\n";
		}cout<<"\n";

		cout<<"\n\ninv_K\n";
		for(int i=0; i<4; i++){
			for(int j=0; j<4; j++){
				cout<<"\t"<< std::setw(5)<<inv_K.operator()(i,j);
			}cout<<"\n";
		}cout<<"\n";

		cout<<"\n\ncam2cam\n";
		for(int i=0; i<4; i++){
			for(int j=0; j<4; j++){
				cout<<"\t"<< std::setw(5)<<cam2cam.operator()(i,j);
			}cout<<"\n";
		}cout<<"\n";

		cout << "\n\ndemo pixel";														// demo pixel transform
		cv::Matx41f pixel((320),(240),(1),(1));
		cv::Matx41f new_pixel;
		for (int i=0; i<4; i++){
			pixel.operator()(3) = i;
			new_pixel = cam2cam * pixel;

			cout<<"\n depth="<<i<<",  pixel=(";
			for (int j=0; j<4; j++){
				cout<< pixel.operator()(j)  << ", ";
			}
			cout<<")";

			cout<<", new_pixel=(";
			for (int j=0; j<3; j++){
				cout<< new_pixel.operator()(j)/new_pixel.operator()(3)  << ", ";
			}
			cout<<"1.00)"<<flush;
		}

		//////// Need: pose2pose  =  keyframe2world^-1  *  world2newpose
		// NB it is the keyframe matricies that must be inverted and stored with the costvol object.
		cout<<"///////////////////////////////////////////////"<<flush;
	}
	if(verbosity>0) cout << "\nupdateCost chk4," << flush;
	assert(baseImage.isContinuous() ); 													// TODO move these assertions to the constructor ?
	//assert(lo.isContinuous() );		//  returns true if the matrix elements are stored continuously without gaps at the end of each row.
	//assert(hi.isContinuous() );		// 
	image = image.reshape(0, rows);									// line 85: rows = image.rows, i.e. num rows in the base image. If the parameter is 0, the number of channels remains the same
	float k2k[16];
	for (int i=0; i<16; i++) { k2k[i] = cam2cam.operator()(i/4, i%4); }

	cvrc.calcCostVol(k2k, image);														// calls calcCostVol(..) #################
	if(verbosity>0) cout << "\nupdateCost chk5_finished\n" << flush;
}

void CostVol::cacheGValues()
{
	if(verbosity>1) {cout<<"\nCostVol::cacheGValues() \nbaseImageGray.empty()="<<baseImageGray.empty()<<" \nbaseImageGray.size="<<baseImageGray.size<<flush;}
	cvrc.cacheGValue2(baseImageGray, theta);
}

void CostVol::updateQD()
{														if(verbosity>1) cout<<"\nupdateQD_chk0, epsilon="<<epsilon<<" theta="<<theta<<flush;
	computeSigmas(epsilon, theta);						if(verbosity>1) cout<<"\nupdateQD_chk1, epsilon="<<epsilon<<" theta="<<theta<<" sigma_q="<<sigma_q<<" sigma_d="<<sigma_d<<flush;
	cvrc.updateQD(epsilon, theta, sigma_q, sigma_d);	if(verbosity>1) cout<<"\nupdateQD_chk3, epsilon="<<epsilon<<" theta="<<theta<<" sigma_q="<<sigma_q<<" sigma_d="<<sigma_d<<flush;
}

bool CostVol::updateA()
{
	if(verbosity>1) cout<<"\nCostVol::updateA "<<flush;
	if (theta < 0.001 && old_theta > 0.001){  cacheGValues(); old_theta=theta; }		// If theta falls below 0.001, then G must be recomputed.
	// bool doneOptimizing = (theta <= thetaMin);
	cvrc.updateA(layers,lambda,theta);
	theta *= thetaStep;
	//return doneOptimizing;
	if(verbosity>1) cout<<"\nCostVol::updateA finished"<<flush;
	return false;
}

void CostVol::writePointCloud(cv::Mat depthMap)
{
	cv::Mat tempImage = 255 * baseImage.reshape(0, rows*cols);

	cv::Mat pointCloud(depthMap.size(), CV_32FC3);
	if(verbosity>0) cout <<"\na)pointCloud.size = " << pointCloud.size() << "\n"<<flush;

	for (int u=0; u<rows ; u++){
		for (int v=0; v<cols ; v++){
			cv::Vec4f homogeneousPoint = inv_K * cv::Vec4f( u, v, 1, depthMap.at<float>(u,v) );
			pointCloud.at<Vec3f>(u,v) = cv::Vec3f(homogeneousPoint[0], homogeneousPoint[1], homogeneousPoint[2]) /homogeneousPoint[3];
		}
	}
	if(verbosity>0) cv::imshow("pointCloud depthmap", pointCloud );
	pointCloud = pointCloud.reshape(0, rows*cols);

	if(verbosity>0) cout << "\nb)pointCloud.size() = " << pointCloud.size() << "\tinv_K.size() = ‘class cv::Matx<float, 4, 4>’\n"<<flush;

	boost::filesystem::path folder_results  =  cvrc.paths.at("amem");
	stringstream ss;
	ss << folder_results.parent_path().string() << "/depthmap.pts"  ;
	if(verbosity>0) cout << "\nDepthmap file : " << ss.str() << "\n" << flush;
	char buf[256];
    sprintf ( buf, "%s", ss.str().data() ); // /depthmap.xyz   folder_results.parent_path().string()
	if(verbosity>0) cout << "buf"<< buf << "\n" << flush;
	FILE* fp = fopen ( buf, "w" );
	if (fp == NULL) {
        cout << "\nvoid CostVol::writePointCloud(cv::Mat depthMap)  Could not open file "<< fp <<"\n"<< std::flush;
        assert(0);
    }
    fprintf(fp, "%u\n", rows*cols);
	for (int i=0; i<rows*cols; i++){		// (x,y,z) coordinates,  "intensity"{0-255},  "color" (rgb){0-255}.
		fprintf(fp, "%f %f %f %i %i %i %i\n", pointCloud.at<Vec3f>(i)[0], pointCloud.at<Vec3f>(i)[1], pointCloud.at<Vec3f>(i)[2], 100, (int)(0.5+tempImage.at<Vec3f>(i)[2]), (int)(0.5+tempImage.at<Vec3f>(i)[1]), (int)(0.5+tempImage.at<Vec3f>(i)[1]) );
	}
	fclose ( fp );
    fflush ( fp );
	if(verbosity>0) cout << "\nwritePointCloud(..) finished\n" << flush;
}

void CostVol::GetResult()
{
	if(verbosity>1) cout<<"\nCostVol::GetResult"<<flush;
	cvrc.ReadOutput(_a.data);
	writePointCloud(_a);
	cvrc.CleanUp();
	if(verbosity>1) cout<<"\nCostVol::GetResult_finished"<<flush;
}
