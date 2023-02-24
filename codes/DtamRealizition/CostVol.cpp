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
	/*
	cout << "CostVol_chk -1\n" << flush;
	cout << "KEY\tPATH\n";															// print the folder paths
    for (auto itr = paths.begin(); itr != paths.end(); ++itr) {
        cout << "First:["<<  itr->first << "]\t:\t Second:" << itr->second << "\n";
    }
	*/
	//cvrc(out_path);   // ? how to instantiate ?
	//cvrc.printPaths();
	cout << "CostVol_chk 0\n" << flush;
	// New code to
	// K, inv_K, pose, inv_pose,
	pose = pose.zeros();
	cout << "CostVol_chk 1\n" << flush;
	for (int i=0; i<9; i++) pose.operator()(i/3,i%3) = R.at<float>(i/3,i%3);
	for (int i=0; i<3; i++) pose.operator()(i,3)     = T.at<float>(i);
	pose.operator()(3,3) = 1;
	cout << "CostVol_chk 2\n" << flush;
	inv_pose = pose.inv();

	cout << "CostVol_chk 3\n" << flush;
	K = K.zeros();														// NB currently "cameraMatrix" found by convertAhandPovRay, called by fileLoader
	for (int i=0; i<9; i++) {
		cout<<"\ni="<<i<<","<<flush;
		cout<<"\tK.operator()(i/3,i%3)="<<K.operator()(i/3,i%3)<<","<<flush;
		cout<<"\tcameraMatrix.at<float>(i/3,i%3)="<<_cameraMatrix.at<float>(i/3,i%3)<<","<<flush;
		K.operator()(i/3,i%3) = _cameraMatrix.at<float>(i/3,i%3);

	} // TODO later, restructure mainloop to build costvol continuously...
	K.operator()(3,3)  = 1;

	cout << "CostVol_chk 4\n" << flush;
	float fx   =  K.operator()(0,0);  cout<<"\nfx="  <<fx;
	float fy   =  K.operator()(1,1);  cout<<"\nfy="  <<fy;
	float skew =  K.operator()(0,1);  cout<<"\nskew="<<skew;
	float cx   =  K.operator()(0,2);  cout<<"\ncx="  <<cx;
	float cy   =  K.operator()(1,2);  cout<<"\ncy="  <<cy;

	cout << "CostVol_chk 5\n" << flush;
	float scalar = 1/(fx*fy);
	inv_K = inv_K.zeros();
	inv_K.operator()(0,0)  = 1.0/fx;  cout<<"\n1.0/fx="<<1.0/fx;
	inv_K.operator()(1,1)  = 1.0/fy;  cout<<"\n1.0/fy="<<1.0/fy;
	inv_K.operator()(2,2)  = 1.0;
	inv_K.operator()(3,3)  = 1.0/(fx*fy);

	inv_K.operator()(0,1)  = -skew/(fx*fy);
	inv_K.operator()(0,2)  = (cy*skew - cx*fy)/(fx*fy);
	inv_K.operator()(1,2)  = (-cy*fx)/(fx*fy);

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

	std::cout << "CostVol_chk 0\n" << std::flush;
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
	cameraMatrix = _cameraMatrix.clone();
    
	// solve projection with
	solveProjection(R, T);
    cout << "CostVol_chk 1.2\n" << flush;
	costdata = Mat::zeros(layers, rows * cols, CV_32FC1);
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
	cvrc.allocatemem( (float*)_qx.data, (float*)_qy.data, (float*)_gx.data, (float*)_gy.data );

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

/*	inv_K computed in CostVol constructor.
	cv::Matx44f inv_K = cv::Matx44f::zeros();                    // = K.inv();  // K^-1 from keyframe. Belongs in constructor and costvol object data
	float scalar = 1/(fx*fy);
	inv_K = inv_K.zeros();
	inv_K.operator()(0,0)  = 1.0/fx;  cout<<"\n1.0/fx="<<1.0/fx;
	inv_K.operator()(1,1)  = 1.0/fy;  cout<<"\n1.0/fy="<<1.0/fy;
	inv_K.operator()(2,2)  = 1.0;
	inv_K.operator()(3,3)  = 1.0/(fx*fy); // ? or just 1 ?

	inv_K.operator()(0,1)  = -skew/(fx*fy);
	inv_K.operator()(0,2)  = (cy*skew - cx*fy)/(fx*fy);
	inv_K.operator()(1,2)  = (-cy*fx)/(fx*fy);
*/
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

	//////// Need: pose2pose  =  keyframe2world^-1  *  world2newpose
	// NB it is the keyframe matricies that must be inverted and stored with the costvol object.
	cout<<"///////////////////////////////////////////////"<<flush;

	/*
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                                                                        //find projection matrix from cost volume to image (3x4)
	Mat cameraMatrixTex(3, 4, CV_32FC1);
    
	cout << "\nupdateCost chk2," << flush;
	cameraMatrixTex = 0.0;
	cameraMatrix.copyTo(cameraMatrixTex(Range(0, 3), Range(0, 3)));
	cameraMatrixTex(Range(0, 2), Range(2, 3)) += 0.5;                   //add 0.5 to x,y out //removing causes crash
	


	std::cout << std::fixed << std::setprecision(2);					//  Inspect values in matricies ///////

	cout<<"\n\nviewMatrixImage\n";
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<viewMatrixImage.operator()(i,j);
		}cout<<"\n";
	}cout<<"\n";


	cout<<"\n\ncameraMatrix\n";
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<cameraMatrix.at<float>(i,j);
		}cout<<"\n";
	}cout<<"\n";

	cout<<"\n\ncameraMatrixTex\n";
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<cameraMatrixTex.at<float>(i,j);
		}cout<<"\n";
	}cout<<"\n";

	cout<<"\n\nprojection\n";
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<projection.at<float>(i,j);
		}cout<<"\n";
	}cout<<"\n";

	Mat inv_proj = projection.inv();
	cout<<"\n\nprojection.inv()\n";
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<inv_proj.at<float>(i,j);
		}cout<<"\n";
	}cout<<"\n";


	cout << "\nupdateCost chk3," << flush;
	Mat imFromWorld = cameraMatrixTex * viewMatrixImage;                //3x4  matrix //////////////////
	Mat imFromCV    = imFromWorld * projection.inv();
    

	//  Inspect values in matricies
	cout<<"\n\nimFromWorld = cameraMatrixTex * viewMatrixImage;\n";
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<imFromWorld.at<float>(i,j);
		}cout<<"\n";
	}cout<<"\n";

	cout<<"\n\nimFromCV = imFromWorld * projection.inv();\n";
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<imFromCV.at<float>(i,j);
		}cout<<"\n";
	}cout<<"\n";
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
*/

	cout << "\nupdateCost chk4," << flush;
	assert(baseImage.isContinuous() );
	assert(lo.isContinuous() );
	assert(hi.isContinuous() );
    
	/*
	// persp matrix //////////
	cout << "\nupdateCost chk5," << flush;
	float *p = (float*)imFromCV.data;
	float persp[12];
	cout<<"\n\nprojection matrix, as floats:\n";
	for (int i = 0; i<12; i++) {	// Load the projection matrix as ana array of 12 floats.
        persp[i] = p[i]; 			// Implicit conversion float->float.
    }
    for(int p_iter=0;p_iter<12;p_iter++){cout<<"p["<<p_iter<<"]="<<p[p_iter]<<"\t\t"; if((p_iter+1)%4==0){cout<<"\n";};}
    cout<<"\n";
	*/

	image = image.reshape(0, rows); // line 85: rows = image.rows, i.e. num rows in the base image. If the parameter is 0, the number of channels remains the same
	/*
	//memcpy(costd, (float*)costdata.data, st);
	//float* hitd = (float*)malloc(st);
	//memcpy(hitd, (float*)hit.data, st);
	*/
	/*
	// camera_params //////
	float camera_params[3];
	//int mat_type = cameraMatrix.type();
	camera_params[0] = cameraMatrix.at<double>(0,0)  ; // focal length, assumes fx=fy
	camera_params[1] = cameraMatrix.at<double>(0,2)  ; // c_x
	camera_params[2] = cameraMatrix.at<double>(1,2)  ; // c_y
	cout<<"\n";
	cout<<"\nfx=camera_params[0]="<<camera_params[0];
	cout<<"\ncx=camera_params[1]="<<camera_params[1];
	cout<<"\ncy=camera_params[2]="<<camera_params[2];
	cout<<"\n"<<flush;
	*/

	/*
	// rt_mtx ///////  R & T  TODO fix this !
	Mat rt_mtx;
	hconcat(R, T, rt_mtx);
	//for (int i = 0; i<12; i++) {	// Load the pose transform matrix as ana array of 12 doubles.
    //    persp[i] = p[i]; 			// Implicit conversion double->float.
    //}

	float rt[12];

	cout<<"\nrt_mtx="<<"\n";
	for (int i=0; i<3; i++){
		for (int j=0; j<4; j++){
			rt[i*3 + j] = rt_mtx.at<double>(i,j);  // implicit doble->float conversion
			cout<<rt_mtx.at<double>(i,j)<<",";
		}cout<<"\n";
	}cout<<"\n"<<flush;
	*/
//	cout<<"\nTemporary early halt.\n"<<flush;   cl_int res = 0;   cvrc.exit_(res); //////////////// Temporary early halt.

/*
	// reproj_rot
	float *reproj_rot;
	//viewMatrixImage

	// reproj_rt
	float *reproj_rt;
*/
	cout << "\nupdateCost chk6," << flush;
	float k2k[16];
	for (int i=0; i<16; i++){ k2k[i] = cam2cam.operator()(i/4, i%4); }
                                                                        // calls calcCostVol(..) ###############################
	// reproj_rot/*camera_params*/, reproj_rt/*persp*//*rt*/
	cvrc.calcCostVol(k2k, baseImage, image, (float*)costdata.data, (float*)hit.data, occlusionThreshold, layers);
	cout << "\nupdateCost chk7_finished\n" << flush;
	/*
    //memcpy(hit.data, hitd, st);
	
    size_t st = rows * cols * layers * sizeof(float);
	float* costd = (float*)malloc(st);
	memcpy(costd ,costdata.data, st);

	double min = 0, max = 0;
	minMaxIdx(costdata, &min, &max);
	*/
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

