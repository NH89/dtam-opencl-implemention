//#pragma once
#ifndef RUNCL_H
#define RUNCL_H

#include <CL/cl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
							// indices for float params passed to __const params_buf
#define PIXELS			0	// TODO Can these be #included from a common header for both host and device code?
#define ROWS			1
#define COLS			2
#define LAYERS			3
#define MAX_INV_DEPTH	4
#define MIN_INV_DEPTH	5
#define INV_DEPTH_STEP	6
#define ALPHA_G			7
#define BETA_G			8	//  __kernel void CacheG4
#define EPSILON 		9	//  __kernel void UpdateQD		// epsilon = 0.1
#define SIGMA_Q 		10									// sigma_q = 0.0559017
#define SIGMA_D 		11
#define THETA			12
#define LAMBDA			13	//   __kernel void UpdateA2
#define SCALE_EAUX		14

#define verbosity		1	// -1= none, 0=errors only, 1=basic, 2=lots.

using namespace std;
class RunCL
{
public:
	std::vector<cl_platform_id> m_platform_ids;
	cl_context			m_context;
	cl_device_id		m_device_id;
	cl_command_queue	m_queue;
	cl_program			m_program;
	cl_kernel			cost_kernel, cache3_kernel, cache4_kernel, updateQD_kernel, updateA_kernel;
	cl_mem				basemem, imgmem, cdatabuf, hdatabuf, k2kbuf, dmem, amem, basegraymem, gxmem, gymem, g1mem, qmem, lomem, himem, param_buf, img_sum_buf;

	size_t  			global_work_size, local_work_size, image_size_bytes;
	bool 				gpu, amdPlatform;
	cl_device_id 		deviceId;
	float 				params[16] = {0};
	int 				width, height, costVolLayers, baseImage_type, count=0, keyFrameCount=0, costVolCount=0, QDcount=0, A_count=0;
	cv::Size 			baseImage_size;
	std::map< std::string, boost::filesystem::path > paths;

	RunCL(boost::filesystem::path out_path);
	void createFolders(boost::filesystem::path out_path); // NB called by RunCL(..) constructor, above.
	void saveCostVols(float max_range);
	void DownloadAndSave(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range );
	void DownloadAndSave_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show );
	void DownloadAndSaveVolume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range );

	void allocatemem(float* gx, float* gy, float* params, int layers, cv::Mat &baseImage, float *cdata, float *hdata, float *img_sum_data); 			/*float *qx, float *qy,*/
	void calcCostVol(float* k2k, cv::Mat &image); 						/*cv::Mat &baseImage,*/ /*float thresh, int layers*/ /*, float *cdata, float *hdata*/
	void cacheGValue2(cv::Mat &bgray, float theta);
	void updateQD(float epsilon, float theta, float sigma_q, float sigma_d);
	void updateA(int layers, float lambda, float theta);

	void CleanUp();
	void exit_(cl_int res);
	~RunCL();

	//void DownloadAndSaveVolume_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show );
	//void initializeCostVol(float* k2k, cv::Mat &baseImage, cv::Mat &image, float *cdata, float *hdata, float thresh, int layers);
	//void initializeAD();

	int  convertToString(const char *filename, std::string& s){
		size_t size;
		char*  str;
		std::fstream f(filename, (std::fstream::in | std::fstream::binary));
		if (f.is_open() ) {
			size_t fileSize;
			f.seekg(0, std::fstream::end);
			size = fileSize = (size_t)f.tellg();
			f.seekg(0, std::fstream::beg);
			str = new char[size + 1];
			if (!str) {
				f.close();
				return 0;
			}
			f.read(str, fileSize);
			f.close();
			str[size] = '\0';
			s = str;
			delete[] str;
			return 0;
		}
		cout << "Error: failed to open file\n:" << filename << endl;
		return 1;
	}

	int waitForEventAndRelease(cl_event *event){
		cout << "\nwaitForEventAndRelease_chk0, event="<<event<<" *event="<<*event << flush;
		cl_int status = CL_SUCCESS;
		status = clWaitForEvents(1, event); if (status != CL_SUCCESS) { cout << "\nclWaitForEvents status=" << status << ", " <<  checkerror(status) <<"\n" << flush; exit_(status); }
		status = clReleaseEvent(*event); 	if (status != CL_SUCCESS) { cout << "\nclReleaseEvent status="  << status << ", " <<  checkerror(status) <<"\n" << flush; exit_(status); }
		return status;
	}

	string checkerror(int input) {
		int errorCode = input;
		switch (errorCode) {
		case -9999:											return "Illegal read or write to a buffer";		// NVidia error code
		case CL_DEVICE_NOT_FOUND:							return "CL_DEVICE_NOT_FOUND";
		case CL_DEVICE_NOT_AVAILABLE:						return "CL_DEVICE_NOT_AVAILABLE";
		case CL_COMPILER_NOT_AVAILABLE:						return "CL_COMPILER_NOT_AVAILABLE";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:				return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case CL_OUT_OF_RESOURCES:							return "CL_OUT_OF_RESOURCES";
		case CL_OUT_OF_HOST_MEMORY:							return "CL_OUT_OF_HOST_MEMORY";
		case CL_PROFILING_INFO_NOT_AVAILABLE:				return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case CL_MEM_COPY_OVERLAP:							return "CL_MEM_COPY_OVERLAP";
		case CL_IMAGE_FORMAT_MISMATCH:						return "CL_IMAGE_FORMAT_MISMATCH";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:					return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case CL_BUILD_PROGRAM_FAILURE:						return "CL_BUILD_PROGRAM_FAILURE";
		case CL_MAP_FAILURE:								return "CL_MAP_FAILURE";
		case CL_MISALIGNED_SUB_BUFFER_OFFSET:				return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:	return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case CL_INVALID_VALUE:								return "CL_INVALID_VALUE";
		case CL_INVALID_DEVICE_TYPE:						return "CL_INVALID_DEVICE_TYPE";
		case CL_INVALID_PLATFORM:							return "CL_INVALID_PLATFORM";
		case CL_INVALID_DEVICE:								return "CL_INVALID_DEVICE";
		case CL_INVALID_CONTEXT:							return "CL_INVALID_CONTEXT";
		case CL_INVALID_QUEUE_PROPERTIES:					return "CL_INVALID_QUEUE_PROPERTIES";
		case CL_INVALID_COMMAND_QUEUE:						return "CL_INVALID_COMMAND_QUEUE";
		case CL_INVALID_HOST_PTR:							return "CL_INVALID_HOST_PTR";
		case CL_INVALID_MEM_OBJECT:							return "CL_INVALID_MEM_OBJECT";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:			return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case CL_INVALID_IMAGE_SIZE:							return "CL_INVALID_IMAGE_SIZE";
		case CL_INVALID_SAMPLER:							return "CL_INVALID_SAMPLER";
		case CL_INVALID_BINARY:								return "CL_INVALID_BINARY";
		case CL_INVALID_BUILD_OPTIONS:						return "CL_INVALID_BUILD_OPTIONS";
		case CL_INVALID_PROGRAM:							return "CL_INVALID_PROGRAM";
		case CL_INVALID_PROGRAM_EXECUTABLE:					return "CL_INVALID_PROGRAM_EXECUTABLE";
		case CL_INVALID_KERNEL_NAME:						return "CL_INVALID_KERNEL_NAME";
		case CL_INVALID_KERNEL_DEFINITION:					return "CL_INVALID_KERNEL_DEFINITION";
		case CL_INVALID_KERNEL:								return "CL_INVALID_KERNEL";
		case CL_INVALID_ARG_INDEX:							return "CL_INVALID_ARG_INDEX";
		case CL_INVALID_ARG_VALUE:							return "CL_INVALID_ARG_VALUE";
		case CL_INVALID_ARG_SIZE:							return "CL_INVALID_ARG_SIZE";
		case CL_INVALID_KERNEL_ARGS:						return "CL_INVALID_KERNEL_ARGS";
		case CL_INVALID_WORK_DIMENSION:						return "CL_INVALID_WORK_DIMENSION";
		case CL_INVALID_WORK_GROUP_SIZE:					return "CL_INVALID_WORK_GROUP_SIZE";
		case CL_INVALID_WORK_ITEM_SIZE:						return "CL_INVALID_WORK_ITEM_SIZE";
		case CL_INVALID_GLOBAL_OFFSET:						return "CL_INVALID_GLOBAL_OFFSET";
		case CL_INVALID_EVENT_WAIT_LIST:					return "CL_INVALID_EVENT_WAIT_LIST";
		case CL_INVALID_EVENT:								return "CL_INVALID_EVENT";
		case CL_INVALID_OPERATION:							return "CL_INVALID_OPERATION";
		case CL_INVALID_GL_OBJECT:							return "CL_INVALID_GL_OBJECT";
		case CL_INVALID_BUFFER_SIZE:						return "CL_INVALID_BUFFER_SIZE";
		case CL_INVALID_MIP_LEVEL:							return "CL_INVALID_MIP_LEVEL";
		case CL_INVALID_GLOBAL_WORK_SIZE:					return "CL_INVALID_GLOBAL_WORK_SIZE";
		case CL_INVALID_DEVICE_QUEUE:						return "CL_INVALID_DEVICE_QUEUE";
		case CL_INVALID_PIPE_SIZE:							return "CL_INVALID_PIPE_SIZE";
		default:											return "unknown error code";
		}
	}

	string checkCVtype(int input) {
		int errorCode = input;
		switch (errorCode) {
			//case	CV_8U:										return "CV_8U";
			case	CV_8UC1:										return "CV_8UC1";
			case	CV_8UC2:										return "CV_8UC2";
			case	CV_8UC3:										return "CV_8UC3";
			case	CV_8UC4:										return "CV_8UC4";

			//case	CV_8S:										return "CV_8S";
			case	CV_8SC1:										return "CV_8SC1";
			case	CV_8SC2:										return "CV_8SC2";
			case	CV_8SC3:										return "CV_8SC3";
			case	CV_8SC4:										return "CV_8SC4";

			//case	CV_16U:										return "CV_16U";
			case	CV_16UC1:										return "CV_16UC1";
			case	CV_16UC2:										return "CV_16UC2";
			case	CV_16UC3:										return "CV_16UC3";
			case	CV_16UC4:										return "CV_16UC4";

			//case	CV_16S:										return "CV_16S";
			case	CV_16SC1:										return "CV_16SC1";
			case	CV_16SC2:										return "CV_16SC2";
			case	CV_16SC3:										return "CV_16SC3";
			case	CV_16SC4:										return "CV_16SC4";

			//case	CV_32S:										return "CV_32S";
			case	CV_32SC1:										return "CV_32SC1";
			case	CV_32SC2:										return "CV_32SC2";
			case	CV_32SC3:										return "CV_32SC3";
			case	CV_32SC4:										return "CV_32SC4";

			//case	CV_32F:										return "CV_32F";
			case	CV_32FC1:										return "CV_32FC1";
			case	CV_32FC2:										return "CV_32FC2";
			case	CV_32FC3:										return "CV_32FC3";
			case	CV_32FC4:										return "CV_32FC4";

			//case	CV_64F:										return "CV_64F";
			case	CV_64FC1:										return "CV_64FC1";
			case	CV_64FC2:										return "CV_64FC2";
			case	CV_64FC3:										return "CV_64FC3";
			case	CV_64FC4:										return "CV_64FC4";

			default:										return "unknown CV_type code";
		}
	}

	void ReadOutput(uchar* outmat) {
		ReadOutput(outmat, amem,  (width * height * sizeof(float)) );
	}

	void ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset=0) {
		cl_event readEvt;
		cl_int status;
		cout<<"\nReadOutput: &outmat="<<&outmat<<", buf_mem="<<buf_mem<<", data_size="<<data_size<<", offset="<<offset<<"\t"<<flush;
		status = clFlush(m_queue);		if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
		status = clFinish(m_queue);		if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="		<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
		status = clEnqueueReadBuffer(m_queue,
											buf_mem,
											CL_FALSE,
											offset,
											data_size,
											outmat,
											0,
											NULL,
											&readEvt);
												if (status != CL_SUCCESS) { cout << "\nclEnqueueReadBuffer(..) status=" << checkerror(status) <<"\n"<<flush; exit_(status);}
		status = clFlush(m_queue);				if (status != CL_SUCCESS) { cout << "\nclFlush(m_queue) status = " 		<< checkerror(status) <<"\n"<<flush; exit_(status);}
		status = clWaitForEvents(1, &readEvt); 	if (status != CL_SUCCESS) { cout << "\nclWaitForEvents status="			<< checkerror(status) <<"\n"<<flush; exit_(status);}
		status = clFinish(m_queue);				if (status != CL_SUCCESS) { cout << "\nclFinish(m_queue)=" 				<< checkerror(status) <<"\n"<<flush; exit_(status);}
	}
};

#endif /*RUNCL_H*/
