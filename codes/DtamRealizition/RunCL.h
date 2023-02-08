#pragma once

//#include <CL/cl.h>  // included via <CL/cl.hpp> & CL/opencl.h
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

using namespace std;
class RunCL
{
public:
	RunCL(boost::filesystem::path out_path);
	
	std::vector<cl_platform_id> m_platform_ids;
	cl_context                  m_context;
	cl_device_id                m_device_id;
	cl_command_queue            m_queue;
	cl_program                  m_program;

	//cl_kernel  cost_kernel, min_kernel, optiQ_kernel, optiD_kernel, optiA_kernel;
	cl_kernel  cost_kernel, cache1_kernel, cache2_kernel, cache3_kernel, cache4_kernel, initializeAD_kernel, updateQ_kernel, updateD_kernel, updateA_kernel;
	cl_mem basemem, imgmem, cdatabuf, hdatabuf, pbuf, dmem, amem; /*, gdmem, gumem, glmem, grmem, qxmem, qymem,*/
	cl_mem basegraymem, gxmem, gymem, g1mem, gqxmem, gqymem, lomem, himem;
	size_t  global_work_size, local_work_size;
	bool gpu;
	bool amdPlatform;     
	cl_device_id deviceId;  
	
	int width, height, costVolLayers;
	int count=0, keyFrameCount=0, costVolCount=0, QDcount=0, Acount=0;

	size_t image_size_bytes;
	cv::Size baseImage_size;
	int baseImage_type;

	std::map< std::string, boost::filesystem::path > paths;
	void createFolders(boost::filesystem::path out_path); // NB called by RunCL(..) constructor, above.

	int  convertToString(const char *filename, std::string& s)
	{
		size_t size;
		char*  str;
		std::fstream f(filename, (std::fstream::in | std::fstream::binary));

		if (f.is_open())
		{
			size_t fileSize;
			f.seekg(0, std::fstream::end);
			size = fileSize = (size_t)f.tellg();
			f.seekg(0, std::fstream::beg);
			str = new char[size + 1];
			if (!str)
			{
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

	void DownloadAndSave(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show);
	//void DownloadAndSave(cl_mem buffer, boost::filesystem::path folder, size_t size);

	void DownloadAndSaveVolume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show );

	void calcCostVol(float* p, cv::Mat &baseImage, cv::Mat &image, float *cdata, float *hdata, float thresh, int layers);
    /*	
	//void minv(float *loInd, float *loVal, int layers);
	//void optiQ(float sigma_q,float epsilon,float denom);
	//void optiD(float sigma_d,float epsilon,float denom, float theta);
	//void optiA(float theta,float ds,float lamda,int l,int layerstep);
    */
	void cacheGValue2(cv::Mat &bgray, float theta);
	void cacheGValue(cv::Mat &bgray);

	void initializeAD();

	void updateQD(float epsilon, float theta, float sigma_q, float sigma_d);
	void updateA(int layers, float lambda, float theta);

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
	
	void allocatemem(float *qx, float *qy, float* gx, float* gy);

	void ReadOutput(uchar* outmat) {  // cvrc.ReadOutput(_a.data, dmem, image_size_bytes );
		// DownloadAndSave(dmem, (keyFrameCount*1000 + costVolCount), paths.at("dmem"),  width * height * sizeof(float), baseImage_size, CV_32FC1, /*show=*/ true );
		ReadOutput(outmat, dmem,  (width * height * sizeof(float)) );  // image_size_bytes
		/*
		cl_event readEvt;
		cl_int status = clEnqueueReadBuffer(m_queue,
											dmem,
											CL_FALSE,
											0,
											width * height * sizeof(float),
											outmat,
											0,
											NULL,
											&readEvt);
		clFlush(m_queue);
		//clFinish(m_queue);
		//clReleaseEvent(readEvt);
		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}
		*/
	}

	void ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset=0) {
		cl_event readEvt;
		cout<<"\nReadOutput: &outmat="<<&outmat<<", buf_mem="<<buf_mem<<", data_size="<<data_size<<", offset="<<offset<<"\t"<<flush;
		cl_int status = clEnqueueReadBuffer(m_queue,
											buf_mem,
											CL_FALSE,
											offset,
											data_size,
											outmat,
											0,
											NULL,
											&readEvt);
		clFlush(m_queue);
		//clWaitForEvents (1, &readEvt);
		clFinish(m_queue);
		//clReleaseEvent(readEvt);
		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\t"<<flush; exit_(status);}
		//cout<<"\nReadOutput finished\n"<<flush;
	}

	void CleanUp();

	void exit_(cl_int res);

	~RunCL();
};

