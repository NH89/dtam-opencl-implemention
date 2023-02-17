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

	projection.create(4, 4, CV_64FC1);											// NB cv::Mat projection; declared in CostVol.h
	projection = 0.0;
	projection(Range(0, 2), Range(0, 3)) += cameraMatrix.rowRange(0, 2);
	//Projection:
	//fx 0  cx 0
	//0  fy cy 0
	//0  0  0  0
	//0  0  0  0

	projection.at<double>(2, 3) = 1.0;
	projection.at<double>(3, 2) = 1.0;
	//Projection: Takes camera coordinates to pixel coordinates:x_px,y_px,1/zc
	//fx 0  cx 0
	//0  fy cy 0
	//0  0  0  1
	//0  0  1  0

	Mat originShift = (Mat)(Mat_<double>(4, 4) << 1.0, 0., 0., 0.,
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
	const cv::Mat& _cameraMatrix) {
    
	assert(R.size() == Size(3, 3));
	assert(R.type() == CV_64FC1);
	assert(T.size() == Size(1, 3));
	assert(T.type() == CV_64FC1);
	assert(_cameraMatrix.size() == Size(3, 3));
	assert(_cameraMatrix.type() == CV_64FC1);
    
	CV_Assert(_cameraMatrix.at<double>(2, 0) == 0.0);
	CV_Assert(_cameraMatrix.at<double>(2, 1) == 0.0);
	CV_Assert(_cameraMatrix.at<double>(2, 2) == 1.0);
}

CostVol::CostVol(Mat image, FrameID _fid, int _layers, float _near,
	float _far, cv::Mat R, cv::Mat T, cv::Mat _cameraMatrix, float occlusionThreshold,
	boost::filesystem::path out_path,
	float initialCost, float initialWeight
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
    
	depthStep    = (near - far) / (layers - 1);
	cameraMatrix = _cameraMatrix.clone();
    





	// solve projection with
	solveProjection(R, T);
    
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
	baseImage = baseImage.reshape(0, rows); 				// redundant, given that rows = baseImage.rows
    
	cout << "CostVol_chk 4\n" << flush;
	cvtColor(baseImage, baseImageGray, CV_RGB2GRAY);
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
                                                                        //find projection matrix from cost volume to image (3x4)
	cout << "\nupdateCost chk1," << flush;
	Mat viewMatrixImage;
	RTToP(R, T, viewMatrixImage);
	Mat cameraMatrixTex(3, 4, CV_64FC1);
    
	cout << "\nupdateCost chk2," << flush;
	cameraMatrixTex = 0.0;
	cameraMatrix.copyTo(cameraMatrixTex(Range(0, 3), Range(0, 3)));
	cameraMatrixTex(Range(0, 2), Range(2, 3)) += 0.5;                   //add 0.5 to x,y out //removing causes crash
	

	//  Inspect values in matricies
	std::cout << std::fixed << std::setprecision(2);

	cout<<"\n\ncameraMatrix\n";
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<cameraMatrix.at<double>(i,j);
		}cout<<"\n";
	}cout<<"\n";

	cout<<"\n\ncameraMatrixTex\n";
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<cameraMatrixTex.at<double>(i,j);
		}cout<<"\n";
	}cout<<"\n";

	cout<<"\n\nprojection\n";
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<projection.at<double>(i,j);
		}cout<<"\n";
	}cout<<"\n";

	Mat inv_proj = projection.inv();
	cout<<"\n\nprojection.inv()\n";
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<inv_proj.at<double>(i,j);
		}cout<<"\n";
	}cout<<"\n";


	cout << "\nupdateCost chk3," << flush;
	Mat imFromWorld = cameraMatrixTex * viewMatrixImage;                //3x4  matrix //////////////////
	Mat imFromCV    = imFromWorld * projection.inv();
    

	//  Inspect values in matricies
	cout<<"\n\nimFromWorld = cameraMatrixTex * viewMatrixImage;\n";
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<imFromWorld.at<double>(i,j);
		}cout<<"\n";
	}cout<<"\n";

	cout<<"\n\nimFromCV = imFromWorld * projection.inv();\n";
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			cout<<"\t"<< std::setw(5)<<imFromCV.at<double>(i,j);
		}cout<<"\n";
	}cout<<"\n";



	cout << "\nupdateCost chk4," << flush;
	assert(baseImage.isContinuous() );
	assert(lo.isContinuous() );
	assert(hi.isContinuous() );
    
	// persp matrix //////////
	cout << "\nupdateCost chk5," << flush;
	double *p = (double*)imFromCV.data;
	float persp[12];
	cout<<"\n\nprojection matrix, as doubles:\n";
	for (int i = 0; i<12; i++) {	// Load the projection matrix as ana array of 12 doubles.
        persp[i] = p[i]; 			// Implicit conversion double->float.
    }
    for(int p_iter=0;p_iter<12;p_iter++){cout<<"p["<<p_iter<<"]="<<p[p_iter]<<"\t\t"; if((p_iter+1)%4==0){cout<<"\n";};}
    cout<<"\n";
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
	cl_int res = 0;
	cvrc.exit_(res); //////////////// Temporary early halt.

	// reproj_rot
	float *reproj_rot;
	//viewMatrixImage

	// reproj_rt
	float *reproj_rt;

	cout << "\nupdateCost chk6," << flush;
                                                                        // calls calcCostVol(..) ###############################
	cvrc.calcCostVol(reproj_rot/*camera_params*/, reproj_rt/*persp*//*rt*/, baseImage, image, (float*)costdata.data, (float*)hit.data, occlusionThreshold, layers);
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

