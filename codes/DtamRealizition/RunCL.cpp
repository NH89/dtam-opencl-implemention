// #include "stdafx.h"  // windows precompiled header file
#include "RunCL.h"

#define SDK_SUCCESS 0
#define SDK_FAILURE 1

using namespace std;

RunCL::RunCL(boost::filesystem::path out_path) // constructor
{
	cout << "\nRunCL_chk -1\n" << flush;
	createFolders(out_path);

	cout << "\nRunCL_chk 0\n" << flush;
	/*Step1: Getting platforms and choose an available one.*////////////////////////////////////////////////////////////
	cl_uint numPlatforms;	//the NO. of platforms
	cl_platform_id platform = NULL;	//the chosen platform
	cl_int	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS){
		cout << "Error: Getting platforms!" << endl;
		exit_(status);//return;
	}
	cout << "numPlatforms = " << numPlatforms << "\n" << flush;

	/*Choose the platform. */
	if (numPlatforms > 0){
		cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));

		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		cout << "platforms[0] = " << platforms[0] << ", \nplatforms[1] = " << platforms[1] << "\n" << flush;

		int n=1; // rx6400=0 or gtx3080=1  ################################ Choose GPU here #### NB Avoid sharing GPU with GUI #######################################
		cout << "\nSelected platform number :"<<n<<"\n"<<flush;

		platform = platforms[n]; //0 or 1 , Need to choose platform and GPU ##################################### Choose platform ############## TODO replace with launch yaml file.
		free(platforms);

		//size_t param_value_size;
		//void* param_value;
		//size_t* param_value_size_ret;
		//cl_int clGetPlatformInfo( platform, CL_PLATFORM_NAME, param_value_size, param_value, param_value_size_ret);
		//cout << "\n CL_PLATFORM_NAME = " param_value << "\n"<<flush;;
	}
	cout << "cl_platform_id platform = " << platform ;

	cout << "\nRunCL_chk 1\n" << flush; /*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/////////////
	cl_uint				numDevices = 0;
	cl_device_id        *devices;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	if (status != CL_SUCCESS) {cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}

	cout << "RunCL_chk 2\n" << flush;
	if (numDevices == 0){	//no GPU available.
		cout << "No GPU device available. Choose CPU as default device." << endl;
		status  = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status  = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
	}else{
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status  = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	}
	cout << "RunCL_chk 3\n" << flush; cout << "cl_device_id  devices = " << devices << "\n" << flush;   // pointer to array of unkown length...
	if (status != CL_SUCCESS) {cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}

	/*Step 3: Create context.*///////////////////////////////////////////////////////////////////////////////////////////
	 cl_context_properties cps[3] ={
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platform,
		0
	};

	cout << "RunCL_chk 4\n" << flush;
	m_context = clCreateContextFromType(
										cps,
										CL_DEVICE_TYPE_GPU,
										NULL,
										NULL,
										&status);
	if (status != 0){cout << " status=" << status << ", " << checkerror(status) << "\n"<< flush; exit_(status);}

	/*Step 4: Creating command queue associate with the context.*/
	deviceId = devices[0];
	cl_command_queue_properties prop[] = { 0 };						//  all queues are in-order by default.
	m_queue = clCreateCommandQueueWithProperties(m_context,
													deviceId,
													prop,
													&status);
	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}

	cout << "RunCL_chk 5\n" << flush; /*Step 5: Create program object *////////////////////////////////////////////////////////////////////////////////////
	const char *filename = "/home/nick/Programming/ComputerVision/DTAM/dtam-opencl-implemention/codes/DtamRealizition/DTAM_kernels2.cl"; //"DTAM_kernels2.cl";
	string sourceStr;
	status = convertToString(filename, sourceStr);
	const char *source = sourceStr.c_str();
	size_t sourceSize[] = { strlen(source) };
	m_program = clCreateProgramWithSource(m_context, 1, &source, sourceSize, NULL);
	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}

	cout << "RunCL_chk 6\n" << flush; /*Step 6: Build program. *///////////////////////////////////////////////////////////////////////////////////////////
	status = clBuildProgram(m_program, 1, devices, NULL, NULL, NULL);
	if (status != CL_SUCCESS){
		printf("\nclBuildProgram failed: %d\n", status);
		char buf[0x10000];
		clGetProgramBuildInfo(m_program, deviceId, CL_PROGRAM_BUILD_LOG, 0x10000, buf, NULL);
		printf("\n%s\n", buf);
		exit_(status);//return;
	}

	cout << "RunCL_chk 7\n" << flush; //*Step 7: Create kernel object. *///////////////////////////////////////////////////////////////////////////////////
	// cost_kernel, cache3_kernel, cache4_kernel, updateQD_kernel, updateA_kernel;
	cost_kernel     = clCreateKernel(m_program, "BuildCostVolume2", NULL);
	cache3_kernel   = clCreateKernel(m_program, "CacheG3", NULL);
	cache4_kernel   = clCreateKernel(m_program, "CacheG4", NULL);
	updateQD_kernel = clCreateKernel(m_program, "UpdateQD", NULL);
	updateA_kernel  = clCreateKernel(m_program, "UpdateA2", NULL);

	//cache1_kernel         = clCreateKernel(m_program, "CacheG1", NULL);
	//cache2_kernel         = clCreateKernel(m_program, "CacheG2", NULL);
	//initializeAD_kernel	= clCreateKernel(m_program, "InitializeAD", NULL);
	//updateQ_kernel        = clCreateKernel(m_program, "UpdateQ", NULL);
	//updateD_kernel        = clCreateKernel(m_program, "UpdateD", NULL);

	cout << "RunCL_chk 8\n" << flush;
	// set device pointers to zero
	basemem=imgmem=cdatabuf=hdatabuf=k2kbuf=dmem=amem=0;//qxmem=qymem=pbuf=kbuf=rtbuf
	basegraymem=gxmem=gymem=g1mem=lomem=himem=0; //gqxmem=gqymem=

	cout << "RunCL_chk 9\n" << flush;
}


void RunCL::createFolders(boost::filesystem::path out_path){
	boost::filesystem::path temp_path = out_path;
																					// Vector of device buffer names
	std::vector<std::string> names = {"basemem","imgmem","cdatabuf","hdatabuf","pbuf","dmem", "amem","basegraymem","gxmem","gymem","g1mem","qmem","lomem","himem"};
	// "qxmem","qymem", "gqxmem","gqymem"
	std::pair<std::string, boost::filesystem::path> tempPair;
	//std::map <std::string, boost::filesystem::path> paths;

	for (std::string key : names){
		temp_path = out_path;
		temp_path += key;
		tempPair = {key, temp_path};
		paths.insert(tempPair);
		boost::filesystem::create_directory(temp_path);
		temp_path += "/png/";
		boost::filesystem::create_directory(temp_path);
	}
	cout << "\nRunCL::createFolders() chk1\n";
    cout << "KEY\tPATH\n";															// print the folder paths
    for (auto itr = paths.begin(); itr != paths.end(); ++itr) {
        cout << "First:["<<  itr->first << "]\t:\t Second:" << itr->second << "\n";
    }
    cout<<"\npaths.at(\"basemem\")="<<paths.at("basemem")<<"\n"<<flush;
}

/*
buffers being saved:

DownloadAndSave:
amem CV_32FC1	max63		amem_update min -ve why ?
qmem		max 1
dmem		max 1

lomem		max 3 ? why
himem		max 1 ? why

basegraymem     max 256
gxmem		max 256
gymem		max 256
g1mem		max 1 ? why

imgmem CV_32FC3
basemem		max(256,256,256)

DownloadAndSaveVolume:
cdatabuf CV_32FC1		max 1 ?
hdatabuf			max 30.001
 */
void RunCL::DownloadAndSave(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range ){
		cout<<"\n\nDownloadAndSave filename = ["<<folder_tiff.filename().string()<<"] folder="<<folder_tiff<<", image_size_bytes="<<image_size_bytes<<", size_mat="<<size_mat<<", type_mat="<<size_mat<<"\t"<<flush;
		cv::Mat temp_mat = cv::Mat::zeros (size_mat, type_mat);									//(int rows, int cols, int type)
		ReadOutput(temp_mat.data, buffer,  image_size_bytes); 									// NB contains elements of type_mat, (CV_32FC1 for most buffers)
		cv::Scalar sum = cv::sum(temp_mat);														// NB always returns a 4 element vector.

		double minVal=1, maxVal=1;
		cv::Point minLoc={0,0}, maxLoc{0,0};
		if (temp_mat.channels()==1) { cv::minMaxLoc(temp_mat, &minVal, &maxVal, &minLoc, &maxLoc); }
		string type_string = checkCVtype(type_mat);

		stringstream ss;
		ss << "/" << folder_tiff.filename().string() << "_" << count <<"_sum"<<sum<<"type_"<<type_string<<"min"<<minVal<<"max"<<maxVal;//<< ".png";//.tiff//.bmp//.pfm//.hdr
		if(show){
			cv::Mat temp;
			temp_mat.convertTo(temp, CV_8U);                             // NB need CV_U8 for imshow(..)
			cv::imshow( ss.str(), temp);
		}
		boost::filesystem::path folder_png = folder_tiff;
		folder_tiff += ss.str();
		folder_tiff += ".tiff";
		folder_png  += "/png/";
		folder_png  += ss.str();
		folder_png  += ".png";

		cout<<"\n\nDownloadAndSave filename = ["<<ss.str()<<"]";

		cv::Mat outMat;

		if (type_mat == CV_32FC1){
				cv::imwrite(folder_tiff.string(), 	(temp_mat/max_range) ); // (temp_mat/(float)maxVal)
				temp_mat *= 256*256/max_range;
				temp_mat.convertTo(outMat, CV_16UC1);
				cv::imwrite(folder_png.string(),	(outMat)	);
		}
}


void RunCL::DownloadAndSave_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show ){
		cout<<"\n\nDownloadAndSave filename = ["<<folder_tiff.filename()<<"] folder="<<folder_tiff<<", image_size_bytes="<<image_size_bytes<<", size_mat="<<size_mat<<", type_mat="<<size_mat<<"\t"<<flush;
		cv::Mat temp_mat = cv::Mat::zeros (size_mat, type_mat);									//(int rows, int cols, int type)
		ReadOutput(temp_mat.data, buffer,  image_size_bytes); 									// NB contains elements of type_mat, (CV_32FC1 for most buffers)

		cv::Scalar sum = cv::sum(temp_mat);														// NB always returns a 4 element vector.
		string type_string = checkCVtype(type_mat);

		double minVal[3]={1,1,1}, maxVal[3]={0,0,0};
		cv::Point minLoc[3]={{0,0},{0,0},{0,0}}, maxLoc[3]={{0,0},{0,0},{0,0}};
		vector<cv::Mat> spl;
		split(temp_mat, spl);                							// process - extract only the correct channel
		double max = 0;
		for (int i =0; i < 3; ++i){
			cv::minMaxLoc(spl[i], &minVal[i], &maxVal[i], &minLoc[i], &maxLoc[i]);
			if (maxVal[i] > max) max = maxVal[i];
		}

		stringstream ss;
		ss << "/" << folder_tiff.filename().string() << "_" << count <<"_sum"<<sum<<"type_"<<type_string<<"min("<<minVal[0] <<","<<minVal[1] <<","<<minVal[2]<<")_max("<<maxVal[0]<<","<<maxVal[1]<<","<<maxVal[2]<<")";
		if(show){
			cv::Mat temp;
			temp_mat.convertTo(temp, CV_8U);                             // NB need CV_U8 for imshow(..)
			cv::imshow( ss.str(), temp);
		}
		boost::filesystem::path folder_png = folder_tiff;
		folder_tiff += ss.str();
		folder_tiff += ".tiff";
		folder_png  += "/png/";
		folder_png  += ss.str();
		folder_png  += ".png";

		cv::Mat outMat;
		if (type_mat == CV_32FC3){
				cv::imwrite(folder_tiff.string(), 	(temp_mat/(256)) );
				temp_mat.convertTo(outMat, CV_8U);
				cv::imwrite(folder_png.string(),	(outMat)	);		/*(temp_mat*(1/maxVal))*/ 	// Has "Grayscale 16-bit gamma integer"
		}
}


void RunCL::DownloadAndSaveVolume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show ){
	cout<<"\n\nDownloadAndSaveVolume, costVolLayers="<<costVolLayers<<", filename = ["<<folder.filename().string()<<"]";
	cout<<"\n folder="<<folder.string()<<",\t image_size_bytes="<<image_size_bytes<<",\t size_mat="<<size_mat<<",\t type_mat="<<size_mat<<"\t"<<flush;
	cv::Mat temp_mat = cv::Mat::zeros (size_mat, type_mat);										//(int rows, int cols, int type)

	for(int i=0; i<costVolLayers; i++){
		cout << "\ncostVolLayers="<<costVolLayers<<", i="<<i<<"\t";
		size_t offset = i * image_size_bytes;
		ReadOutput(temp_mat.data, buffer,  image_size_bytes, offset);

		cv::Scalar sum = cv::sum(temp_mat);
		double minVal=1, maxVal=1;
		cv::Point minLoc={0,0}, maxLoc{0,0};
		if (type_mat == CV_32FC1) cv::minMaxLoc(temp_mat, &minVal, &maxVal, &minLoc, &maxLoc);

		boost::filesystem::path new_filepath = folder;
		boost::filesystem::path folder_png   = folder;

		string type_string = checkCVtype(type_mat);
		stringstream ss;
		ss << "/"<< folder.filename().string() << "_" << count << "_layer"<< i <<"_sum"<<sum<<"type_"<<type_string<< "min"<<minVal<<"max"<<maxVal;
		if(show){
			cv::Mat temp;
			temp_mat.convertTo(temp, CV_8U);                             		// NB need CV_U8 for imshow(..)
			cv::imshow(ss.str(), temp);
		}
		new_filepath += ss.str();
		new_filepath += ".tiff";
		folder_png += "/png/";
		folder_png += ss.str();
		folder_png += ".png";

		cout << "\nnew_filepath.string() = "<<new_filepath.string() <<"\n";
		cv::Mat outMat;
		if (type_mat == CV_32FC1) {
			cv::imwrite(new_filepath.string(), 	(temp_mat*(1/maxVal))     );	// NB .tiff values from 0.0 to 1.0     // Has "Grayscale 32bit gamma floating point"
			temp_mat = temp_mat*(256.0*256.0/maxVal);
			temp_mat.convertTo(outMat, CV_16UC1);
			cv::imwrite(folder_png.string(), 	(outMat) );
		}
	}
}


void RunCL::calcCostVol(float* k2k, cv::Mat &baseImage, cv::Mat &image, float *cdata /*default all=3.0*/, float *hdata, float thresh, int layers) // TODO should be split initializer and updater functions.
{
	cout << "\ncalcCostVol chk0," << flush;
	stringstream ss;
	ss << "calcCostVol";
	size_t old_image_size_bytes = image_size_bytes;
	image_size_bytes 			= image.total() * image.elemSize() ;  			// num array elements * bytes per element all channels
	costVolCount++;
	keyFrameCount++;
	if (costVolCount > 1 && old_image_size_bytes!=image_size_bytes){cout << "\n\nMismatch 'image_size_bytes' !!!\n"<<flush; exit_(0); }
	cl_int status, res;
	cl_event writeEvt;
	costVolLayers 	= layers;
	int pixelSize 	= baseImage.channels();
	baseImage_size 	= baseImage.size();
	baseImage_type 	= baseImage.type();											// pixelSize *  sizeof(baseImage.type())
	size_t baseImage_size_bytes = baseImage.total() * baseImage.elemSize() ;	// NB must equal width * height * pixelSize,

	cout <<"\ncalcCostVol chk1, baseImage_size_bytes="<< baseImage_size_bytes << ", baseImage.total()=" << baseImage.total() << ", sizeof(float)="<< sizeof(float)<<flush;
	cout <<"\nbaseImage.elemSize()="<< baseImage.elemSize()<<", baseImage.elemSize1()="<<baseImage.elemSize1()<<flush;
	cout <<"\nbaseImage.type()="<< baseImage.type() <<", sizeof(baseImage.type())="<< sizeof(baseImage.type())<<flush;
	cout<<"\n";
	cout <<"\ncalcCostVol chk1, image_size_bytes="<< image_size_bytes << ", image.total()=" << image.total() << ", sizeof(float)="<< sizeof(float)<<flush;
	cout <<"\nimage.elemSize()="<< image.elemSize()<<", image.elemSize1()="<<image.elemSize1()<<flush;
	cout <<"\nImage.type()="<< image.type() <<", sizeof(Image.type())="<< sizeof(image.type())<<flush;

	if (basemem == 0) {// basemem _not_ yet allocated. ///////////////////////////////////////////////////////
		cout << "\ncalcCostVol chk2," << flush;
		imgmem   = clCreateBuffer(m_context, CL_MEM_READ_ONLY 						 ,    image_size_bytes, 0, &res);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		basemem  = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,    image_size_bytes, 0, &res);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		k2kbuf   = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR	 ,  16 * sizeof(float),k2k,&res); 		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		cdatabuf = clCreateBuffer(m_context, CL_MEM_READ_WRITE ,   width * height * layers * sizeof(float), 0, &res);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		hdatabuf = clCreateBuffer(m_context, CL_MEM_READ_WRITE ,   width * height * layers * sizeof(float), 0, &res);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		if (0 == basemem || 0 == imgmem) return;

		status = clEnqueueWriteBuffer(m_queue, 	// WriteBuffer basemem ##########
			basemem,
			CL_FALSE,
			0,
			width * height * sizeof(float) *3,
			baseImage.data,
			0,
			NULL,
			&writeEvt);
		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}

		cout << "\ncalcCostVol chk2.1\n" << flush;
		cv::Mat temp_mat(baseImage.size(), baseImage.type());					// Debug img chk

		clFlush(m_queue);
		status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}

		DownloadAndSave_3Channel(basemem, ss.str(), paths.at("basemem"), image_size_bytes, baseImage.size(), baseImage.type(), false );
		cout << "\ncalcCostVol chk2.2\n" << flush;

		status = clEnqueueWriteBuffer(m_queue, // WriteBuffer imgmem ###########
			imgmem,
			CL_FALSE,
			0,
			image_size_bytes,
			image.data,
			0,
			NULL,
			&writeEvt);
		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}

		cout << "\nimgmem = "<< imgmem <<flush;
		cout << "\nimgmem image_size_bytes= "<< image_size_bytes <<flush;
		cout << "\nwidth * height * pixelSize = " << width * height * pixelSize <<flush;
		cout << "\nimage.size=="<<image.size<<", width="<<width<<", height="<<height <<flush;
		ReadOutput(temp_mat.data, imgmem,  image_size_bytes);					// Debug img chk


		clFlush(m_queue);
		status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}

		status = clEnqueueWriteBuffer(m_queue, // WriteBuffer cdatabuf #########
			cdatabuf,
			CL_FALSE,
			0,
			width * height * layers * sizeof(float),
			cdata,																// default all elements = 3.0, set in CostVol.h
			0,
			NULL,
			&writeEvt);
		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}

		status = clEnqueueWriteBuffer(m_queue, // WriteBuffer hdatabuf #########
			hdatabuf,
			CL_FALSE,
			0,
			width * height * layers * sizeof(float),
			hdata,
			0,
			NULL,
			&writeEvt);
		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}


		status = clEnqueueWriteBuffer(m_queue, // WriteBuffer pbuf #############
			k2kbuf,
			CL_FALSE,
			0,
			16 * sizeof(float),
			k2k, //////////////////////////////////////  NB "k2k" = cam2cam pixel transform
			0,
			NULL,
			&writeEvt);
		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}


		cout << "\ncalcCostVol chk5," << flush;
		status = clFlush(m_queue); 						if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}
		status = waitForEventAndRelease(&writeEvt); 	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}

		cout << "\ncalcCostVol chk6," << flush;
		int layerstep = width * height;
		// Get the maximum work group size for executing the kernel on the device ////////////////////////////////////////////////////////////////////////
		// From https://github.com/rsnemmen/OpenCL-examples/blob/e2c34f1dfefbd265cfb607c2dd6c82c799eb322a/square_array/square.c
		status = clGetKernelWorkGroupInfo(cost_kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_work_size), &local_work_size, NULL);
		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}

		// Number of total work items, calculated here after 1st image is loaded &=> know the size.
		// NB localSize must be devisor
		// NB global_work_size must be a whole number of "Preferred work group size multiple" for Nvidia.
		global_work_size = ceil((float)layerstep/(float)local_work_size) * local_work_size;

		cout<<"\nglobal_work_size="<<global_work_size<<", local_work_size="<<local_work_size<<", deviceId="<<deviceId<<"\n"<<flush;
		cout<<"\nlayerstep=width*height="<<width<<"*"<<height<<"="<<layerstep<<",\tsizeof(layerstep)="<< sizeof(layerstep) <<",\tsizeof(int)="<< sizeof(int) <<flush;

		// set kernelArg
		res = clSetKernelArg(cost_kernel, 0, sizeof(cl_mem),  &k2kbuf);		if(res!=CL_SUCCESS){cout<<"\nk2kbuf res = "   <<checkerror(res)<<"\n"<<flush;exit_(res);} // k
		res = clSetKernelArg(cost_kernel, 1, sizeof(cl_mem),  &basemem);	if(res!=CL_SUCCESS){cout<<"\nbasemem res= "   <<checkerror(res)<<"\n"<<flush;exit_(res);} // base
		res = clSetKernelArg(cost_kernel, 2, sizeof(cl_mem),  &imgmem);		if(res!=CL_SUCCESS){cout<<"\nimgmem res = "   <<checkerror(res)<<"\n"<<flush;exit_(res);} // img
		res = clSetKernelArg(cost_kernel, 3, sizeof(cl_mem),  &cdatabuf);	if(res!=CL_SUCCESS){cout<<"\ncdatabuf res = " <<checkerror(res)<<"\n"<<flush;exit_(res);} // cdata
		res = clSetKernelArg(cost_kernel, 4, sizeof(cl_mem),  &hdatabuf);	if(res!=CL_SUCCESS){cout<<"\nhdatabuf res = " <<checkerror(res)<<"\n"<<flush;exit_(res);} // hdata
		/*
		res = clSetKernelArg(cost_kernel, 5, sizeof(int),     &layerstep);	if(res!=CL_SUCCESS){cout<<"\nlayerstep res = "<<checkerror(res)<<"\n"<<flush;exit_(res);} // layerstep
		res = clSetKernelArg(cost_kernel, 6, sizeof(float),   &thresh);		if(res!=CL_SUCCESS){cout<<"\nthresh res = "   <<checkerror(res)<<"\n"<<flush;exit_(res);} // weight
		res = clSetKernelArg(cost_kernel, 7, sizeof(int),     &width);		if(res!=CL_SUCCESS){cout<<"\nwidth res = "    <<checkerror(res)<<"\n"<<flush;exit_(res);} // cols
		*/
		res = clSetKernelArg(cost_kernel, 5, sizeof(cl_mem),  &lomem);		if(res!=CL_SUCCESS){cout<<"\nlomem res = "    <<checkerror(res)<<"\n"<<flush;exit_(res);} // lo
		res = clSetKernelArg(cost_kernel, 6, sizeof(cl_mem),  &himem);		if(res!=CL_SUCCESS){cout<<"\nhimem res = "    <<checkerror(res)<<"\n"<<flush;exit_(res);} // hi
		res = clSetKernelArg(cost_kernel, 7, sizeof(cl_mem), &amem);		if(res!=CL_SUCCESS){cout<<"\namem res = "     <<checkerror(res)<<"\n"<<flush;exit_(res);} // a
		res = clSetKernelArg(cost_kernel, 8, sizeof(cl_mem), &dmem);		if(res!=CL_SUCCESS){cout<<"\ndmem res = "     <<checkerror(res)<<"\n"<<flush;exit_(res);} // d
		/*
		res = clSetKernelArg(cost_kernel, 12, sizeof(int),    &layers);		if(res!=CL_SUCCESS){cout<<"\nlayers res = "   <<checkerror(res)<<"\n"<<flush;exit_(res);} // layers
		*/
		res = clSetKernelArg(cost_kernel, 9, sizeof(cl_mem), &param_buf);	if(res!=CL_SUCCESS){cout<<"\nparam_buf res = "     <<checkerror(res)<<"\n"<<flush;exit_(res);} // param_buf

		cout<<"\n\nKernelArgs:";
		//cout<<"\npbuf="<<rtbuf;			//0  "rt" rotation & translation, camera pose transform matrix, holds an array of 12 images, each as as 2D float array.
		cout<<"\n\npose transformation matrix, as floats:\n";
		for(int p_iter=0;p_iter<16;p_iter++){  cout<<"rt["<<p_iter<<"]="<<k2k[p_iter]<<"\t\t";  if((p_iter+1)%4==0){cout<<"\n";}  }
		cout<<"\n";
		cout<<"\nk2kbuf="<<k2kbuf;			//0
		//cout<<", focal length="<<k[0]<<", c_x="<<k[1]<<", c_y="<<k[2];
		//cout<<"\nbasemem="<<basemem ;		//1  baseImage.data  // the keyframe
		//cout<<"\nimgmem="<<imgmem ;   	//2  image.data		 // the new frame, being added to costvol
		//cout<<"\ncdatabuf="<<cdatabuf ;	//3  cdata=costdata.data, initialCost, default = 3.0,			initialized when costvol constructed
		//cout<<"\nhdatabuf="<<hdatabuf ;	//4  hdata=hit.data,  hit = initialWeight, default = .001		initialized when costvol constructed
		cout<<"\nlayerstep="<<layerstep ;	//5  global_work_size=layerstep=width*height=640*480=307200
		cout<<"\nthresh="<<thresh ;			//6
		cout<<"\nwidth="<<width ;			//7
		//cout<<"\nlomem="<<lomem ;			//8  lomem  = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res); // used between kernels.
		//cout<<"\nhimem="<<himem ;			//9  himem  = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);
		//cout<<"\namem="<<amem ;			//10 amem   = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);
		//cout<<"\ndmem="<<dmem ;			//11 dmem   = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);
		cout<<"\nlayers="<<layers ;			//12 layers = 32
		cout<<"\n"<<flush;

		local_work_size=32; // trial for nvidia

		cout << "\n\ncalcCostVol chk7," << flush;
		cl_event ev;
		cout<<"\n\nclEnqueueNDRangeKernel(";
		cout<<"\ncommand_queue="<<m_queue;
		cout<<"\nkernel=cost_kernel="<<cost_kernel;
		cout<<"\nwork_dim="<<1;
		cout<<"\nglobal_work_offset="<<0;
		cout<<"\nglobal_work_size="<<global_work_size;
		cout<<"\nlocal_work_size="<<local_work_size/*"NULL"*//*0*/;
		cout<<"\nnum_events_in_wait_list"<<0;
		cout<<"\nevent_wait_list=NULL";
		cout<<"\nevent="<<ev;
		cout<<"\n)"<<flush;

		status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
		status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}

		res = clEnqueueNDRangeKernel(m_queue, cost_kernel, 1, 0, &global_work_size, &local_work_size, 0, NULL, &ev); 		/// call cost_kernel ############
		status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}

		if (res != CL_SUCCESS)		{ cout << "\nclEnqueueNDRangeKernel res = " << checkerror(res) <<"\n"<<flush; exit_(res);}
		if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
		cout << "\n\ncalcCostVol chk7.5" << flush;
	}
	else // if the device buffers already exist ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	{											////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		cout << "\n\ncalcCostVol chk8," << flush;
		cout << "\nm_queue=" <<m_queue<<
				"\n imgmem=" <<imgmem<<
				"\n CL_FALSE=" <<CL_FALSE<<
				"\n width*height*pixelSize=" <<width*height*pixelSize<<
				//" image.data=" <<image.data<<
				"\n &writeEvt=" <<&writeEvt<<flush;

		status = clEnqueueWriteBuffer(m_queue,
			imgmem,
			CL_FALSE,
			0,
			image_size_bytes,//width * height * pixelSize,
			image.data,
			0,
			NULL,
			&writeEvt);
		if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer imgmem status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
		cout << "\nimgmem = "<< imgmem <<flush;
		cout << "\nimgmem image_size_bytes= "<< image_size_bytes <<flush;
		cout << "\nwidth * height * pixelSize = " << width * height * pixelSize <<flush;
		cout << "\nimage.size=="<<image.size<<", width="<<width<<", height="<<height <<flush;

																			// flush command queue to device 			// wait for device to complete execution
		status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) clEnqueueWriteBuffer imgmem status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
		status = clWaitForEvents(1, &writeEvt); if (status != CL_SUCCESS)	{ cout << "\nclWaitForEvents clEnqueueWriteBuffer imgmem status = "<<status<<"  "<<checkerror(status) <<"\n"<<flush; exit_(status); }

		status = clEnqueueWriteBuffer(m_queue, // WriteBuffer pbuf #############
			k2kbuf,
			CL_FALSE,
			0,
			16 * sizeof(float),
			k2k, //////////////////////////////////////  NB "k2k" = cam2cam pixel transform
			0,
			NULL,
			&writeEvt);
		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}

		cout << "\n\ncalcCostVol chk9, m_queue="<<m_queue<< flush;			// flush command queue to device			// clFinish needs command_queue, not event list.
		status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
		status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}

		int layerstep = width * height;

		cout << "\ncalcCostVol chk10," << flush;	// set kernelArg
		res = clSetKernelArg(cost_kernel, 0, sizeof(cl_mem), &k2kbuf);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 2, sizeof(cl_mem), &imgmem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		/*
		res = clSetKernelArg(cost_kernel, 5, sizeof(int),    &layerstep);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 6, sizeof(float),  &thresh);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 7, sizeof(int),    &width);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 12, sizeof(int),   &layers);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		*/

		cout<<"\n\npose transformation matrix, as floats:\n";
		for(int p_iter=0;p_iter<16;p_iter++)  { cout<<"rt["<<p_iter<<"]="<<k2k[p_iter]<<"\t\t";   if((p_iter+1)%4==0){cout<<"\n"; };  }
		cout<<"\n";
		cout<<"\nk2kbuf="<<k2kbuf;

		cout << "\ncalcCostVol chk11," << flush;
		cl_event ev;
		cout<<"\n\nclEnqueueNDRangeKernel(";
		cout<<"\ncommand_queue="<<m_queue;
		cout<<"\nkernel=cost_kernel="<<cost_kernel;
		cout<<"\nwork_dim="<<1;
		cout<<"\nglobal_work_offset="<<"NULL";
		cout<<"\nglobal_work_size="<<global_work_size;
		cout<<"\nlocal_work_size="<<local_work_size;
		cout<<"\nnum_events_in_wait_list"<<0;
		cout<<"\nevent_wait_list=NULL";
		cout<<"\nevent="<<ev;
		cout<<"\n)"<<flush;

		status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
		status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}

		res    = clEnqueueNDRangeKernel(m_queue, cost_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev); //1, NULL, &global_work_size, &local_work_size, 0, NULL, &ev); ####### cost_kernel #######
		status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
		status = clWaitForEvents (1, &ev);		if (status != CL_SUCCESS)	{ cout << "\nclWaitForEventsh(1, &ev)="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
		status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}

		if (res != CL_SUCCESS)	{ cout << "\nclEnqueueNDRangeKernel res = " << checkerror(res) <<"\n"<<flush; exit_(res);}

	} // End of long "if(first image){..}else{..}"///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	cout << "\ncalcCostVol chk12," << flush;

	// Save buffers to file
	clFlush(m_queue);
	ss << (keyFrameCount*1000 + costVolCount);

	DownloadAndSave_3Channel(imgmem, ss.str(), paths.at("imgmem"), width * height * sizeof(float)*3, baseImage_size,   CV_32FC3, 		 /*show=*/ false ); //
	DownloadAndSave(lomem,  ss.str(), paths.at("lomem"),  width * height * sizeof(float),   baseImage_size,   CV_32FC1, 		 /*show=*/ false , 3);
	DownloadAndSave(himem,  ss.str(), paths.at("himem"),  width * height * sizeof(float),   baseImage_size,   CV_32FC1, 		 /*show=*/ false , 1);
	DownloadAndSave(amem,   ss.str(), paths.at("amem"),   width * height * sizeof(float),   baseImage_size,   CV_32FC1, 		 /*show=*/ false , 0.000512 ); //64  NB 64*inv_d_step
	DownloadAndSave(dmem,   ss.str(), paths.at("dmem"),   width * height * sizeof(float),   baseImage_size,   CV_32FC1, 		 /*show=*/ false , 0.000512 ); //64

	clFlush(m_queue);
	status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
	cout <<"\ncostVolCount="<<costVolCount;
	cout << "\ncalcCostVol chk13_finished\n" << flush;

}// End of RunCL::calcCostVol(...) ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void RunCL::saveCostVols()
{
	cout<<"\nsaveCostVols: Calling DownloadAndSaveVolume";
	stringstream ss;
	ss << "saveCostVols";
	ss << (keyFrameCount*1000 + costVolCount);
	cl_int status;

	DownloadAndSaveVolume(cdatabuf, ss.str(), paths.at("cdatabuf"), width * height * sizeof(float), baseImage_size, CV_32FC1,  false );
	DownloadAndSaveVolume(hdatabuf, ss.str(), paths.at("hdatabuf"), width * height * sizeof(float), baseImage_size, CV_32FC1,  false );

	clFlush(m_queue);
	status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
	cout <<"\ncostVolCount="<<costVolCount;
	cout << "\ncalcCostVol chk13_finished\n" << flush;
}


void RunCL::allocatemem(float *qx,float *qy, float* gx, float* gy, float* params)
{
	cout << "RunCL::allocatemem_chk0" << flush;
	cl_int res;
	dmem   = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	amem   = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	gxmem  = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	gymem  = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	qmem   = clCreateBuffer(m_context, CL_MEM_READ_WRITE , 2*width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	g1mem  = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	lomem  = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	himem  = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	//__constant float* params) //9 pixels, rows, cols, layers, max_inv_depth, min_inv_depth, inv_d_step, threshold
	param_buf = clCreateBuffer(m_context, CL_MEM_READ_ONLY , 16 * sizeof(float), 			0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	cout << "dmem = " << dmem << endl;
	cout << "amem = " << amem << endl;
	cout << "gxmem = " << gxmem << endl;
	cout << "gymem = " << gymem << endl;
	cout << "qmem = " << qmem << endl;
	cout << "g1mem = " << g1mem << endl;
	cout << "lomem = " << lomem << endl;
	cout << "himem = " << himem << endl;
	cout << " " << endl;
	cout << "RunCL::allocatemem_chk1\n" << flush;
	cl_int status;
	cl_event writeEvt;
	status = clEnqueueWriteBuffer(m_queue, // WriteBuffer gxmem ##########
		gxmem,
		CL_FALSE,
		0,
		width * height * sizeof(float),
		gx,
		0,
		NULL,
		&writeEvt);
	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl; exit_(status);}
	else{cout << "allocatemem_chk1.3\n" << flush;}

	status = clEnqueueWriteBuffer(m_queue, // WriteBuffer gymem ##########
		gymem,
		CL_FALSE,
		0,
		width * height * sizeof(float),
		gy,
		0,
		NULL,
		&writeEvt);
	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.4\n" << endl; exit_(status);}
	else{cout << "allocatemem_chk1.4\n" << flush;}

	status = clFlush(m_queue); 						if (status != CL_SUCCESS)	{ cout << "\nclFlush status = " << status << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = waitForEventAndRelease(&writeEvt); 	if (status != CL_SUCCESS)	{ cout << "\nwaitForEventAndRelease status = " << status << checkerror(status) <<"\n"<<flush; exit_(status);}

	status = clEnqueueWriteBuffer(m_queue, // WriteBuffer param_buf ##########
		param_buf,
		CL_FALSE,
		0,
		16 * sizeof(float),
		params,
		0,
		NULL,
		&writeEvt);
	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.4\n" << endl; exit_(status);}
	else{cout << "allocatemem_chk1.4\n" << flush;}

	status = clFlush(m_queue); 						if (status != CL_SUCCESS)	{ cout << "\nclFlush status = " << status << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = waitForEventAndRelease(&writeEvt); 	if (status != CL_SUCCESS)	{ cout << "\nwaitForEventAndRelease status = " << status << checkerror(status) <<"\n"<<flush; exit_(status);}


	cout << "RunCL::allocatemem_finished\n\n" << flush;
}// End of RunCL::allocatemem(..) ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RunCL::~RunCL() // destructor
{
	cl_int status;
	status = clReleaseKernel(cost_kernel);      if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseKernel(cache3_kernel);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseKernel(cache4_kernel);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseKernel(updateQD_kernel);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseKernel(updateA_kernel);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }

	status = clReleaseProgram(m_program);		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseCommandQueue(m_queue);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseContext(m_context);		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
}

void RunCL::CleanUp()
{
	cout<<"\nRunCL::CleanUp_chk0"<<flush;
	cl_int status;
	status = clReleaseMemObject(basemem);	if (status != CL_SUCCESS)	{ cout << "\nbasemem  status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
	status = clReleaseMemObject(imgmem);	if (status != CL_SUCCESS)	{ cout << "\nimgmem   status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.2"<<flush;
	status = clReleaseMemObject(cdatabuf);	if (status != CL_SUCCESS)	{ cout << "\ncdatabuf status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.3"<<flush;
	status = clReleaseMemObject(hdatabuf);	if (status != CL_SUCCESS)	{ cout << "\nhdatabuf status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.4"<<flush;
	//status = clReleaseMemObject(pbuf);		if (status != CL_SUCCESS)	{ cout << "\npbuf     status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.5"<<flush;
	//status = clReleaseMemObject(kbuf);		if (status != CL_SUCCESS)	{ cout << "\nkbuf     status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.5"<<flush;
	//status = clReleaseMemObject(rtbuf);		if (status != CL_SUCCESS)	{ cout << "\nrtbuf    status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.6"<<flush;
	status = clReleaseMemObject(k2kbuf);	if (status != CL_SUCCESS)	{ cout << "\nk2kbuf   status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.5"<<flush;
	status = clReleaseMemObject(qmem);		if (status != CL_SUCCESS)	{ cout << "\ndmem     status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.7"<<flush;
	status = clReleaseMemObject(dmem);		if (status != CL_SUCCESS)	{ cout << "\ndmem     status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.8"<<flush;
	status = clReleaseMemObject(amem);		if (status != CL_SUCCESS)	{ cout << "\namem     status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.9"<<flush;
	status = clReleaseMemObject(lomem);		if (status != CL_SUCCESS)	{ cout << "\nlomem    status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.10"<<flush;
	status = clReleaseMemObject(himem);		if (status != CL_SUCCESS)	{ cout << "\nhimem    status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.11"<<flush;
	cout<<"\nRunCL::CleanUp_chk1_finished"<<flush;
}

void RunCL::exit_(cl_int res)
{
	CleanUp();
	//~RunCL(); Never need to call a destructor manually.
	exit(res);
}

void RunCL::cacheGValue2(cv::Mat &bgray, float theta) // TODO combine cache3_kernel & cache4_kernel
{
	cout<<"\ncacheGValue_chk0"<<flush;
	stringstream ss;
	ss << "cacheGValue2";
	cl_int 		status, res;
	cl_event 	ev;
	size_t 		bgraySize_bytes = bgray.total() * bgray.elemSize();
	float 		beta;
	if (theta>0.001) beta=0.001; else beta=0.0001;

	if (basegraymem == 0) {
		cv::Mat temp;
		bgray.convertTo(temp, CV_8U);                             // NB need CV_U8 for imshow(..)
		//cv::imshow("bgray", bgray);
		cv::imshow("cacheGValue2 bgray CV_8U temp", temp);        // shows correctly

		basegraymem = clCreateBuffer(m_context, CL_MEM_READ_ONLY , bgraySize_bytes, 0, &res);
		status 		= clEnqueueWriteBuffer(m_queue, basegraymem, CL_FALSE, 0, width * height * sizeof(float), bgray.data, 0, NULL, &ev);// WriteBuffer basegraymem ##########
		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}
		clFlush(m_queue); clFinish(m_queue); if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}

		cout<<"\ncacheGValue_chk0.1"<<flush;
		cv::Mat temp_mat = cv::Mat::zeros (baseImage_size, CV_32FC1);
		ReadOutput(temp_mat.data, basegraymem,  width * height * sizeof(float) );
		temp_mat.convertTo(temp_mat, CV_8U);
		cv::imshow("cacheGValue2 basegraymem CV_8U temp", temp_mat);  // does not show correctly !
	}
	cout<<"\ncacheGValue_chk1"<<flush;
	res = clSetKernelArg(cache3_kernel, 0, sizeof(cl_mem), &basegraymem);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cache3_kernel, 1, sizeof(cl_mem), &gxmem);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cache3_kernel, 2, sizeof(cl_mem), &gymem);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cache3_kernel, 3, sizeof(int),    &width);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cache3_kernel, 4, sizeof(int),    &height);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	clFlush(m_queue); clFinish(m_queue); if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	res = clEnqueueNDRangeKernel(m_queue, cache3_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);				// run cache1_kernel  aka CacheG1(..) ##########
	clFlush(m_queue); clFinish(m_queue); if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}

	ss << (keyFrameCount);
	DownloadAndSave(basegraymem, ss.str(), paths.at("basegraymem"), width * height * sizeof(float) , baseImage_size, CV_32FC1, 		 true, 256);
	DownloadAndSave(gxmem,		 ss.str(), paths.at("gxmem"), 		width * height * sizeof(float) , baseImage_size, CV_32FC1, 		 true, 256);
	DownloadAndSave(gymem, 		 ss.str(), paths.at("gymem"), 		width * height * sizeof(float) , baseImage_size, CV_32FC1, 		 true, 256);
	clFlush(m_queue); clFinish(m_queue);

	cout<<"\ncacheGValue_chk2"<<flush;
	res = clSetKernelArg(cache4_kernel, 0, sizeof(cl_mem), &g1mem);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cache4_kernel, 1, sizeof(cl_mem), &gxmem);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cache4_kernel, 2, sizeof(cl_mem), &gymem);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cache4_kernel, 3, sizeof(int),    &width);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cache4_kernel, 4, sizeof(int),    &height);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cache4_kernel, 5, sizeof(float),  &beta);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	res = clEnqueueNDRangeKernel(m_queue, cache4_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);				// run cache1_kernel  aka CacheG1(..) ##########
	clFlush(m_queue); clFinish(m_queue); if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}

	DownloadAndSave(g1mem, 		 ss.str(), paths.at("g1mem"), 		width * height * sizeof(float), baseImage_size, CV_32FC1, 		 true, 1);

	clFlush(m_queue); clFinish(m_queue);
	keyFrameCount++;
	cout<<"\ncacheGValue_chk3_finished"<<flush;
}


void RunCL::updateQD(float epsilon, float theta, float sigma_q, float sigma_d)
{
	cout<<"\nRunCL::updateQD_chk0"<<flush;
	stringstream ss;
	ss << "updateQD";
	cl_int status;
	cl_int res;
	cl_event ev;
	res = clSetKernelArg(updateQD_kernel, 0, sizeof(cl_mem), &g1mem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateQD_kernel, 1, sizeof(cl_mem), &qmem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateQD_kernel, 2, sizeof(cl_mem), &dmem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateQD_kernel, 3, sizeof(cl_mem), &amem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateQD_kernel, 4, sizeof(float),  &epsilon);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//Huber parameter
	res = clSetKernelArg(updateQD_kernel, 5, sizeof(float),  &sigma_q);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateQD_kernel, 6, sizeof(float),  &sigma_d);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateQD_kernel, 7, sizeof(float),  &theta);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);} //Coupling parameter
	res = clSetKernelArg(updateQD_kernel, 8, sizeof(int),    &width);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateQD_kernel, 9, sizeof(int),    &height);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	cout<<"\nRunCL::updateQD_chk1"<<flush;
	res = clEnqueueNDRangeKernel(m_queue, updateQD_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev); 			// run updateQD_kernel  aka UpdateQD(..) ####################
	if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status = clFlush(m_queue);				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clWaitForEvents (1, &ev);		if (status != CL_SUCCESS)	{ cout << "\nclWaitForEventsh(1, &ev)="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue);				if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="		<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}

	cout<<"\nRunCL::updateQD_chk2"<<flush;
	size_t st  = width * height * sizeof(float);
	QDcount++;
	int this_count = count + QDcount;
	ss << this_count;
	cv::Size q_size( baseImage_size.width, 2* baseImage_size.height ); // double sized for qx and qy.

	DownloadAndSave(qmem,   ss.str(), paths.at("qmem"),    2*width * height * sizeof(float), q_size        , CV_32FC1, /*show=*/ false , 1);// 64);//2*640*480);//
	DownloadAndSave(dmem,   ss.str(), paths.at("dmem"),    width * height * sizeof(float)  , baseImage_size, CV_32FC1, /*show=*/ false , 0.000512 ); //64
	clFlush(m_queue);
	clFinish(m_queue);
	cout<<"\nRunCL::updateQD_chk3_finished\n"<<flush;
}


void RunCL::updateA(int layers, float lambda,float theta)
{
	stringstream ss;
	ss << "updateA";
	cl_int status;
	cl_int res;
	cl_event ev;
	res = clSetKernelArg(updateA_kernel, 0, sizeof(cl_mem), &cdatabuf); if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateA_kernel, 1, sizeof(cl_mem), &amem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateA_kernel, 2, sizeof(cl_mem), &dmem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateA_kernel, 3, sizeof(cl_mem), &lomem);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateA_kernel, 4, sizeof(cl_mem), &himem);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	//res = clSetKernelArg(cost_kernel, 8, sizeof(cl_mem),  &lomem);		if(res!=CL_SUCCESS){cout<<"\nlomem res = "    <<checkerror(res)<<"\n"<<flush;exit_(res);} // lo
	//res = clSetKernelArg(cost_kernel, 9, sizeof(cl_mem),  &himem);		if(res!=CL_SUCCESS){cout<<"\nhimem res = "    <<checkerror(res)<<"\n"<<flush;exit_(res);} // hi

	res = clSetKernelArg(updateA_kernel, 5, sizeof(int),    &layers);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateA_kernel, 6, sizeof(int),    &width);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateA_kernel, 7, sizeof(int),    &height);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateA_kernel, 8, sizeof(float),  &lambda);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateA_kernel, 9, sizeof(float),  &theta);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}

	res = clEnqueueNDRangeKernel(m_queue, updateA_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev); 			// run updateA_kernel  aka UpdateA(..) ####################
	if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status = clFlush(m_queue);				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clWaitForEvents (1, &ev);		if (status != CL_SUCCESS)	{ cout << "\nclWaitForEventsh(1, &ev)="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue);				if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="		<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}

	cout<<"\nRunCL::updateA_chk1"<<flush;
	count = keyFrameCount*1000000 + Acount*1000 + 999;
	Acount++;
	if(Acount%1==0){
		ss << count << "_theta"<<theta<<"_";;
		cv::Size q_size( baseImage_size.width, 2* baseImage_size.height ); // double sized for qx and qy.
		//DownloadAndSaveVolume(cdatabuf, ss.str(), paths.at("cdatabuf"), width * height * sizeof(float), baseImage_size, CV_32FC1,  false );
		DownloadAndSave(amem,   ss.str(), paths.at("amem"),    width * height * sizeof(float), baseImage_size, CV_32FC1,  false , 0.000512 ); //64
		DownloadAndSave(dmem,   ss.str(), paths.at("dmem"),    width * height * sizeof(float), baseImage_size, CV_32FC1,  false , 0.000512 ); //64
		DownloadAndSave(qmem,   ss.str(), paths.at("qmem"),  2*width * height * sizeof(float), q_size        , CV_32FC1,  false , 0.000008);//1); inv_d_step
		clFlush(m_queue);
		clFinish(m_queue);
	}
	cout<<"\nRunCL::updateA_chk2_finished"<<flush;
}
