//#include "stdafx.h"     // windows precompiled header file
#include "CostVol.h"
//#include <opencv2/core/operations.hpp>
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
	//P:4x4 rigid transformation taking points from world to the camera frame
	//r1 r2 r3 tx
	//r4 r5 r6 ty
	//r7 r8 r9 tz
	//0  0  0  0

	//Camera:
	//fx 0  cx
	//0  fy cy
	//0  0  1

	projection.create(4, 4, CV_32FC1);											// NB cv::Mat projection; declared in CostVol.h
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

	Mat originShift = (Mat)(Mat_<float>(4, 4) << 1.0, 0., 0., 0.,
		0.,  1.0,  0.,   0.,
		0.,  0.,   1.0,  -far,
		0.,  0.,   0.,   1.0);

	projection = originShift*projection;	//put the origin at 1/z_ from_camera_center = far.   NB main() line 113, costvol constructor, far=0.0 .
	projection.row(2) /= depthStep;			//stretch inverse depth so now   x_cam,y_cam,z_cam-->x_cv_px, y_cv_px , [1/z_from_camera_center - far]_px
	projection = projection*P;				//projection now goes     x_world,y_world,z_world -->x_cv_px, y_cv_px , [1/z_from_camera_center - far]_px

							  // exit(0);
}

void CostVol::checkInputs(
    const cv::Mat& R, 
    const cv::Mat& T,
	const cv::Mat& _cameraMatrix)
{
	std::cout<<"\ncheckInputs\n"<< std::flush;
	assert(R.size() == Size(3, 3));
	assert(R.type() == CV_32FC1);
	assert(T.size() == Size(1, 3));
	assert(T.type() == CV_32FC1);
	assert(_cameraMatrix.size() == Size(3, 3));
	assert(_cameraMatrix.type() == CV_32FC1);
    
	// print cameraMatrix
	std::cout<<"\ncameraMatrix.rows="<<cameraMatrix.rows<<",  cameraMatrix.cols="<<cameraMatrix.cols<<"\n"<< std::flush;   // empty Mat why ?

	for (int i=0; i<cameraMatrix.rows; i++){
		for (int j=0; j<cameraMatrix.cols; j++){
			std::cout<<", "<<cameraMatrix.at<float>(i,j)<< std::flush;
		}std::cout<<"\n"<< std::flush;
	}std::cout<<"\n"<< std::flush;

	CV_Assert(_cameraMatrix.at<float>(2, 0) == 0.0);
	CV_Assert(_cameraMatrix.at<float>(2, 1) == 0.0);
	CV_Assert(_cameraMatrix.at<float>(2, 2) == 1.0);
}



CostVol::CostVol(
	Mat image,
	FrameID 	_fid,
	int 		_layers,
	float 		_near,
	float 		_far,
	cv::Mat 	R,
	cv::Mat 	T,
	cv::Mat 	_cameraMatrix,
	float 		occlusionThreshold,
	boost::filesystem::path 	out_path,
	float 		initialCost,
	float 		initialWeight
) // NB defaults: float initialCost = 3.0, float initialWeight = .001
	:
	R(R), T(T), occlusionThreshold(occlusionThreshold), initialWeight(initialWeight), cvrc(out_path) // constructors for member classes //
{
	cout << "CostVol_chk 0\n" << flush;
	// New code to
	// K, inv_K, pose, inv_pose,
	/*
	 *  Inverse of a transformation matrix:
	 * see http://www.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche0053.html#:~:text=and%20translation%20matrix)-,Inverse%20matrix%20of%20transformation%20matrix%20(rotation%20and%20translation%20matrix),R%7Ct%5D%20transformation%20matrix.&text=The%20inverse%20of%20transformation%20matrix%20%5BR%7Ct%5D%20is%20%5B,%2D%20R%5ET%20t%5D.
	 *
	 *   {     |   }-1       {       |        }
	 *   {  R  | t }     =   {  R^T  |-R^T .t }
	 *   {_____|___}         {_______|________}
	 *   {0 0 0| 1 }         {0  0  0|    1   }
	 *
	 */
	pose = pose.zeros();
	cout << "CostVol_chk 1\n" << flush;
	for (int i=0; i<9; i++) pose.operator()(i/3,i%3) = R.at<float>(i/3,i%3);
	for (int i=0; i<3; i++) pose.operator()(i,3)     = T.at<float>(i);
	pose.operator()(3,3) = 1;
	cout << "CostVol_chk 2\n" << flush;
	//inv_pose = pose.inv();                // opencv mtx::inv() may not be efficient or reliable, depends on choice of decomposition method.
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			inv_pose.operator()(i,j) = pose.operator()(j,i);
		}
	}
	cv::Mat inv_T = -R.t()*T;
	for (int i=0;i<3;i++){
		inv_pose.operator()(i,3)=inv_T.at<float>(i);
	}
	for (int i=0; i<4; i++) inv_pose.operator()(3,i) = pose.operator()(3,i);

	// Verify inv_pose:////////////////////////////////////////////////////
	cout<<"\n\ninv_pose\n";
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<inv_pose.operator()(i,j);
		}cout<<"\n";
	}cout<<"\n";

	cout<<"\n\ntest_pose inversion: pose * inv_pose;\n";
	cv::Matx44f test_pose = pose * inv_pose;
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<test_pose.operator()(i,j);
		}cout<<"\n";
	}cout<<"\n";

	cout<<"\n\ntest_pose inversion: inv_pose * pose;\n";
	/*cv::Matx44f*/ test_pose = inv_pose * pose;
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<test_pose.operator()(i,j);
		}cout<<"\n";
	}cout<<"\n";
	////////////////////////////////////////////////////////////////////////

	cout << "\n\nCostVol_chk 3\n" << flush;
	K = K.zeros();											// NB currently "cameraMatrix" found by convertAhandPovRay, called by fileLoader
	for (int i=0; i<9; i++) {
		cout<<"\ni="<<i<<","<<flush;
		cout<<"\tK.operator()(i/3,i%3)="<<K.operator()(i/3,i%3)<<","<<flush;
		cout<<"\tcameraMatrix.at<float>(i/3,i%3)="<<_cameraMatrix.at<float>(i/3,i%3)<<","<<flush;
		K.operator()(i/3,i%3) = _cameraMatrix.at<float>(i/3,i%3);

	} // TODO later, restructure mainloop to build costvol continuously...
	K.operator()(3,3)  = 1;

	cout << "\n\nCostVol_chk 4\n" << flush;
	float fx   =  K.operator()(0,0);  cout<<"\nfx="  <<fx;
	float fy   =  K.operator()(1,1);  cout<<"\nfy="  <<fy;
	float skew =  K.operator()(0,1);  cout<<"\nskew="<<skew;
	float cx   =  K.operator()(0,2);  cout<<"\ncx="  <<cx;
	float cy   =  K.operator()(1,2);  cout<<"\ncy="  <<cy;

	cout << "\n\nCostVol_chk 5\n" << flush;
	///////////////////////////////////////////////////////////////////// Inverse camera intrinsic matrix, see:
	// https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/#:~:text=Inverse,lines%20from%20the%20camera%20center.
	float scalar = 1/(fx*fy);
	inv_K = inv_K.zeros();
	inv_K.operator()(0,0)  = 1.0/fx;  cout<<"\n1.0/fx="<<1.0/fx;
	inv_K.operator()(1,1)  = 1.0/fy;  cout<<"\n1.0/fy="<<1.0/fy;
	inv_K.operator()(2,2)  = 1.0;
	inv_K.operator()(3,3)  = 1.0;// 1.0/(fx*fy);

	inv_K.operator()(0,1)  = -skew/(fx*fy);
	inv_K.operator()(0,2)  = (cy*skew - cx*fy)/(fx*fy);
	inv_K.operator()(1,2)  = -cy/fy;


	// Verify inv_K:
	cout<<"\n\ntest_camera_intrinsic_matrix inversion\n";
	cv::Matx44f test_K = inv_K * K;
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<test_K.operator()(i,j);
		}cout<<"\n";
	}cout<<"\n";
	////////////////////////////////////////////////////////////////////////


	std::cout << std::fixed << std::setprecision(-1);				//  Inspect values in matricies ///////
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


	/////////////////////////////////////////////////////////////////////////////

	std::cout << "\n\nCostVol_chk 0\n" << std::flush;
	//For performance reasons, OpenDTAM only supports multiple of 32 image sizes with cols >= 64
	CV_Assert(image.rows % 32 == 0 && image.cols % 32 == 0 && image.cols >= 64);
	//     CV_Assert(_layers>=8);

	cout << "CostVol_chk 1\n" << flush;
	checkInputs(R, T, _cameraMatrix);
	fid    = _fid;
	rows   = image.rows;
	cols   = image.cols;
	layers = _layers;
	near   = _near;
	far    = _far;
    cout << "CostVol_chk 1.1\n" << flush;
	depthStep    = (near - far) / (layers - 1);


	//float params[16] = {0};
	cvrc.params[PIXELS] 			= rows*cols;
	cvrc.params[ROWS] 			= rows;
	cvrc.params[COLS] 			= cols;
	cvrc.params[LAYERS] 			= layers;
	cvrc.params[MAX_INV_DEPTH] 	= near;
	cvrc.params[MIN_INV_DEPTH] 	= far;
	cvrc.params[INV_DEPTH_STEP] 	= (cvrc.params[MAX_INV_DEPTH] - cvrc.params[MIN_INV_DEPTH])/(cvrc.params[LAYERS] -1);


	cameraMatrix = _cameraMatrix.clone();

	// solve projection with
	solveProjection(R, T);
    cout << "CostVol_chk 1.2\n" << flush;
	costdata = Mat::ones(layers, rows * cols, CV_32FC1);//zeros
	costdata = initialCost;
    
	hit      = Mat::zeros(layers, rows * cols, CV_32FC1);
	hit      = initialWeight;
	
	FLATALLOC(lo);
	FLATALLOC(hi);
	FLATALLOC(_a);
	FLATALLOC(_d);
	FLATALLOC(_gx);
	FLATALLOC(_gy);
	FLATALLOC(_qx);
    FLATALLOC(_qy);
    FLATALLOC(_g1);
    
	_qx = _qy   = 0;
	_gx = _gy   = 1;
	cvrc.width  = cols;
	cvrc.height = rows;

	cout << "CostVol_chk 2\n" <<flush;
	cvrc.allocatemem( (float*)_qx.data, (float*)_qy.data, (float*)_gx.data, (float*)_gy.data, cvrc.params);

	cout << "CostVol_chk 3\n" << flush;
	image.copyTo(baseImage);
	/*
	Mat temp;
    image.convertTo(temp, CV_8U);                             // NB need CV_U8 for imshow(..)
	cv::imshow("base image CostVol constructor", temp);
	*/
	baseImage = baseImage.reshape(0, rows); 				// redundant, given that rows = baseImage.rows
    
	cout << "CostVol_chk 4\n" << flush;
	cvtColor(baseImage, baseImageGray, CV_RGB2GRAY);
	/*
	cv::Mat temp;
	baseImageGray.convertTo(temp, CV_8U);                             // NB need CV_U8 for imshow(..)
	//cv::imshow("bgray", temp);
	imshow("costVol baseImageGray CV_8U", temp );
	*/
	baseImageGray = baseImageGray.reshape(0, rows);			// baseImageGray used by CostVol::cacheGValues( cvrc.cacheGValue (baseImageGray));


	
	cout << "CostVol_chk 5\n" << flush;
    count      = 0;
	float off  = layers / 32;
	thetaStart = 0.2;		//200.0*off;					// Why ? DTAM_Mapping has Theta_start = 0.2
	thetaMin   = 1.0e-4;	//1.0*off;
	thetaStep  = .97;
	epsilon    = 1.0e-4;	//.1*off;
	lambda     = 1.0;		//.001 / off;
	theta      = thetaStart;
	old_theta  = theta;

	QDruncount = 0;
	Aruncount  = 0;

	alloced = 0;
	cachedG = 0;
	dInited = 0;

	// lambda, theta, sigma_d, sigma_q set by CostVol::computeSigmas(..) in CostVol.h, called in CostVol::updateQD() below.
	computeSigmas(epsilon, theta);
	cvrc.params[ALPHA_G]			=  0.015;		///  __kernel void CacheG4
	cvrc.params[BETA_G]			=  1.5;
	cvrc.params[EPSILON]			=  0.1;			///  __kernel void UpdateQD		// epsilon = 0.1
	cvrc.params[SIGMA_Q]			=  0.0559017;									// sigma_q = 0.0559017
	cvrc.params[SIGMA_D]			=  sigma_d;
	cvrc.params[THETA]			=  theta;
	cvrc.params[LAMBDA]			=  lambda;		///   __kernel void UpdateA2
	cvrc.params[SCALE_EAUX]		=  10000;// from DTAM_Mapping input/json/icl_numin.json    //1.0;
	cout << "CostVol_chk 6\n" << flush;
}


void CostVol::updateCost(const Mat& _image, const cv::Mat& R, const cv::Mat& T) 
{
	cout << "\nupdateCost chk0," << flush;
	Mat image;
	_image.copyTo(image);

	// assemble 4x4 cam2cam homogeneous reprojection matrix:   cam2cam = K(4x4) * RT(4x4) * k^-1(4x4)
	// NB If K changes, then : K from current frame (to be sampled), and K^-1 from keyframe.

	cout << "\nupdateCost chk1," << flush;
	cout<<"\ncameraMatrix.size="<<cameraMatrix.size<<"\n"<<flush;   // R.size=3 x 3,	T.size=3 x 1

	cv::Matx44f K = cv::Matx44f::zeros();//    = cameraMatrix;			// NB currently "cameraMatrix" found by convertAhandPovRay, called by fileLoader
	for (int i=0; i<9; i++) K.operator()(i/3,i%3) = cameraMatrix.at<float>(i/3,i%3);// TODO later, restructure mainloop to build costvol continuously...
	K.operator()(3,3)  = 1;

	cout<<"\n\n'K' camera intrinsic matrix\n";
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<K.operator()(i,j);
		}cout<<"\n";
	}cout<<"\n";

	std::cout << std::fixed << std::setprecision(-1);
	//https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/#:~:text=Inverse,lines%20from%20the%20camera%20center.
	float fx   =  K.operator()(0,0);  cout<<"\nfx="  <<fx;
	float fy   =  K.operator()(1,1);  cout<<"\nfy="  <<fy;
	float skew =  K.operator()(0,1);  cout<<"\nskew="<<skew;
	float cx   =  K.operator()(0,2);  cout<<"\ncx="  <<cx;
	float cy   =  K.operator()(1,2);  cout<<"\ncy="  <<cy;


	cv::Matx44f poseTransform = cv::Matx44f::zeros();
	cout<<"\nR.size="<<R.size<<",\tT.size="<<T.size<<"\n"<<flush;   // R.size=3 x 3,	T.size=3 x 1

	for (int i=0; i<9; i++) poseTransform.operator()(i/3,i%3) = R.at<float>(i/3,i%3);
	for (int i=0; i<3; i++) poseTransform.operator()(i,3)     = T.at<float>(i);			// why is T so large ?
	poseTransform.operator()(3,3) = 1;

	std::cout << std::fixed << std::setprecision(2);				//  Inspect values in matricies ///////
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

	cv::Matx44f cam2cam = K * poseTransform * inv_pose *  inv_K;	// cam2cam pixel transform, NB requires Pixel=(u,v,1,1/z)^T
	cout<<"\n\ncam2cam\n";
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<cam2cam.operator()(i,j);
		}cout<<"\n";
	}cout<<"\n";

	// demo pixel transform
	cv::Matx41f pixel((320),(240),(1),(1));
	cv::Matx41f new_pixel;

	cout << "\n\ndemo pixel";
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

	cout << "\nupdateCost chk4," << flush;
	assert(baseImage.isContinuous() );
	assert(lo.isContinuous() );
	assert(hi.isContinuous() );
    
	image = image.reshape(0, rows); // line 85: rows = image.rows, i.e. num rows in the base image. If the parameter is 0, the number of channels remains the same

	cout << "\nupdateCost chk6," << flush;
	float k2k[16];
	for (int i=0; i<16; i++){ k2k[i] = cam2cam.operator()(i/4, i%4); }
                                                                        // calls calcCostVol(..) ###############################
	cvrc.calcCostVol(k2k, baseImage, image, (float*)costdata.data, (float*)hit.data, occlusionThreshold, layers);
	cout << "\nupdateCost chk7_finished\n" << flush;
}


void CostVol::cacheGValues()
{
	cout<<"\nCostVol::cacheGValues()"<<flush;
	cout<<"\nbaseImageGray.empty()="<<baseImageGray.empty()<<flush;
	cout<<"\nbaseImageGray.size="<<baseImageGray.size<<flush;

	cvrc.cacheGValue2(baseImageGray, theta);
}

/*
void CostVol::initializeAD()
{
	cout<<"\nCostVol::initializeAD()"<<flush;
	cvrc.initializeAD();
}
*/

void CostVol::updateQD()
{
	cout<<"\nupdateQD_chk0, epsilon="<<epsilon<<" theta="<<theta<<flush; // but if theta falls below 0.001, then G must be recomputed.
	computeSigmas(epsilon, theta);

	cout<<"\nupdateQD_chk1, epsilon="<<epsilon<<" theta="<<theta<<" sigma_q="<<sigma_q<<" sigma_d="<<sigma_d<<flush;
	cvrc.updateQD(epsilon, theta, sigma_q, sigma_d);

	cout<<"\nupdateQD_chk3, epsilon="<<epsilon<<" theta="<<theta<<" sigma_q="<<sigma_q<<" sigma_d="<<sigma_d<<flush;
}


bool CostVol::updateA()
{
	if (theta < 0.001 && old_theta > 0.001){cacheGValues(); old_theta=theta;}

	cout<<"\nCostVol::updateA_chk0, "<<flush;
	bool doneOptimizing = theta <= thetaMin;

	cout<<"\nCostVol::updateA_chk1, "<<flush;
	cvrc.updateA(layers,lambda,theta);
	
	cout<<"\nCostVol::updateA_chk2, "<<flush;
	theta *= thetaStep;

	return doneOptimizing;
}


void CostVol::GetResult()
{
	cout<<"\nCostVol::GetResult_chk0"<<flush;
	cvrc.ReadOutput(_a.data); // (float*)

	// DownloadAndSave(dmem,	  (count ), paths.at("dmem"), 	  width * height * sizeof(float), baseImage_size, CV_32FC1, /*show=*/ false );
	// void RunCL::DownloadAndSave(cl_mem buffer,  int count,  boost::filesystem::path folder,  size_t image_size_bytes,  cv::Size size_mat,  int type_mat,  bool show )
	// ReadOutput(temp_mat.data, buffer,  image_size_bytes);
	// cvrc.ReadOutput(_a.data, dmem, image_size_bytes );

	cout<<"\nCostVol::GetResult_chk1"<<flush;
	cvrc.CleanUp();

	cout<<"\nCostVol::GetResult_chk2_finished"<<flush;
}

