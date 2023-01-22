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
		int n=0; // 0 or 1
		cout << "\nSelected platform number :"<<n<<"\n"<<flush;
		platform = platforms[n]; //0 or 1 , Need to choose platform and GPU ##################################### Choose platform ############## TODO replace with launch yaml file.
		free(platforms);
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
	cl_command_queue_properties prop[] = { 0 };
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
	cost_kernel = clCreateKernel(m_program, "BuildCostVolume", NULL);
    /*
	//min_kernel   = clCreateKernel(m_program, "CostMin", NULL);
	//optiQ_kernel = clCreateKernel(m_program, "OptimizeQ", NULL);
	//optiD_kernel = clCreateKernel(m_program, "OptimizeD", NULL);
	//optiA_kernel = clCreateKernel(m_program, "OptimizeA", NULL);
    */
	cache1_kernel  = clCreateKernel(m_program, "CacheG1", NULL);
	cache2_kernel  = clCreateKernel(m_program, "CacheG2", NULL);
	updateQ_kernel = clCreateKernel(m_program, "UpdateQ", NULL);
	updateD_kernel = clCreateKernel(m_program, "UpdateD", NULL);
	updateA_kernel = clCreateKernel(m_program, "UpdateA", NULL);

	cout << "RunCL_chk 8\n" << flush;
	// set device pointers to zero
	basemem=imgmem=cdatabuf=hdatabuf=pbuf=qxmem=qymem=dmem=amem=0;
	basegraymem=gxmem=gymem=g1mem=gqxmem=gqymem=lomem=himem=0;

	cout << "RunCL_chk 9\n" << flush;
}


void RunCL::createFolders(boost::filesystem::path out_path){
	boost::filesystem::path temp_path = out_path;
																					// Vector of device buffer names
	std::vector<std::string> names = {"basemem","imgmem","cdatabuf","hdatabuf","pbuf","qxmem","qymem","dmem", "amem","basegraymem","gxmem","gymem","g1mem","gqxmem","gqymem","lomem","himem"};

	std::pair<std::string, boost::filesystem::path> tempPair;
	//std::map <std::string, boost::filesystem::path> paths;

	for (std::string key : names){
		temp_path = out_path;
		temp_path += key;
		tempPair = {key, temp_path};
		paths.insert(tempPair);
		boost::filesystem::create_directory(temp_path);
	}
	cout << "\nRunCL::createFolders() chk1\n";
    cout << "KEY\tPATH\n";															// print the folder paths
    for (auto itr = paths.begin(); itr != paths.end(); ++itr) {
        cout << "First:["<<  itr->first << "]\t:\t Second:" << itr->second << "\n";
    }
    cout<<"\npaths.at(\"basemem\")="<<paths.at("basemem")<<"\n"<<flush;
}


void RunCL::DownloadAndSave(cl_mem buffer, int count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat ){
		cv::Mat temp_mat(size_mat, type_mat);
		ReadOutput(temp_mat.data, basemem,  image_size_bytes);
		cout<<"\nDownloadAndSave: filename = ["<<folder.filename()<<"] folder = ["<<folder<<"]\n"<<flush;
		stringstream ss;
		ss << "/" << folder.filename().string() << "_" << count << ".bmp";
		cv::imshow(ss.str(), temp_mat);
		folder += ss.str();
		cv::imwrite(folder.string(), temp_mat);
}


void RunCL::calcCostVol(float* p, cv::Mat &baseImage, cv::Mat &image, float *cdata, float *hdata, float thresh, int layers)
{
	cout << "\ncalcCostVol chk0," << flush;
	cl_int status;
	cl_int res;
	cl_event writeEvt;

	int pixelSize = baseImage.channels();
	size_t image_size_bytes = baseImage.total() * baseImage.elemSize() ;		// NB must equal width * height * pixelSize, //  pixelSize *  sizeof(baseImage.type())

	cout <<"\ncalcCostVol chk1, image_size_bytes="<< image_size_bytes << ", baseImage.total()=" << baseImage.total() << ", sizeof(float)="<< sizeof(float)<<flush;
	cout <<"\nbaseImage.elemSize()="<< baseImage.elemSize()<<", baseImage.elemSize1()="<<baseImage.elemSize1()<<flush;
	cout <<"\nbaseImage.type()="<< baseImage.type() <<", sizeof(baseImage.type())="<< sizeof(baseImage.type())<<flush;
	cout <<"\nimage.total()="<< image.total() <<", image.elemSize()="<< image.elemSize()<<flush;

	if (basemem == 0) {// basemem _not_ yet allocated. ///////////////////////////////////////////////////////
		cout << "\ncalcCostVol chk2," << flush;
		imgmem 	 = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, image_size_bytes, image.ptr(), &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		pbuf 	 = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 12 * sizeof(float), p, &res); 			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		basemem  = clCreateBuffer(m_context, CL_MEM_READ_ONLY , image_size_bytes, 0, &res);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		cdatabuf = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * layers * sizeof(float), 0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		hdatabuf = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * layers * sizeof(float), 0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		if (0 == basemem || 0 == imgmem) return;

		status = clEnqueueWriteBuffer(m_queue, // WriteBuffer basemem ##########
			basemem,
			CL_FALSE,
			0,
			image_size_bytes,								// = baseImage.total() * baseImage.elemSize() ; 	//wrong:width * height * pixelSize ,
			baseImage.data,
			0,
			NULL,
			&writeEvt);
		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}

		/// Debug img chk
		cv::Mat temp_mat(baseImage.size(), baseImage.type());
		ReadOutput(temp_mat.data, basemem,  image_size_bytes);
		cv::imshow("basemem_downloaded", temp_mat);
		//cv::waitKey(100);
		clFlush(m_queue);
		DownloadAndSave(basemem, /*count*/ 0, paths.at("basemem"), image_size_bytes, baseImage.size(), baseImage.type() );
	//void RunCL::DownloadAndSave(cl_mem buffer, int count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat )
	//	DownloadAndSave( basemem, paths.at("basemem"), image_size_bytes, baseImage.size(), baseImage.type() );

		status = clEnqueueWriteBuffer(m_queue, // WriteBuffer imgmem ###########
			imgmem,
			CL_FALSE,
			0,
			image_size_bytes,								// = baseImage.total() * baseImage.elemSize() ;		// wrong:width * height * pixelSize,
			image.data,
			0,
			NULL,
			&writeEvt);
		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}
		cout << "\nimgmem = "<< imgmem <<flush;
		cout << "\nimgmem image_size_bytes= "<< image_size_bytes <<flush;
		cout << "\nwidth * height * pixelSize = " << width * height * pixelSize <<flush;
		cout << "\nimage.size=="<<image.size<<", width="<<width<<", height="<<height <<flush;

		/// Debug img chk
		ReadOutput(temp_mat.data, imgmem,  image_size_bytes);
		cv::imshow("imgmem_downloaded", temp_mat);
		//::waitKey(-1);

		status = clEnqueueWriteBuffer(m_queue, // WriteBuffer cdatabuf #########
			cdatabuf,
			CL_FALSE,
			0,
			width * height * layers * sizeof(float),
			cdata,
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
			pbuf,
			CL_FALSE,
			0,
			12 * sizeof(float),
			p, //////////////////////////////////////  NB "p" = projection matrix as an array of 12 floats
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
		global_work_size = ceil((float)layerstep/(float)local_work_size);
		global_work_size = ceil((float)global_work_size/(float)local_work_size) * local_work_size;
		cout<<"\nglobal_work_size="<<global_work_size<<", local_work_size="<<local_work_size<<", deviceId="<<deviceId<<"\n"<<flush;
		cout<<"\nlayerstep=width*height="<<width<<"*"<<height<<"="<<layerstep<<flush;

		// set kernelArg
		res = clSetKernelArg(cost_kernel, 0, sizeof(cl_mem),  &pbuf);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 1, sizeof(cl_mem),  &basemem);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 2, sizeof(cl_mem),  &imgmem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 3, sizeof(cl_mem),  &cdatabuf);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 4, sizeof(cl_mem),  &hdatabuf);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 5, sizeof(int),     &layerstep);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 6, sizeof(float),   &thresh);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 7, sizeof(int),     &width);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 8, sizeof(cl_mem),  &lomem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 9, sizeof(cl_mem),  &himem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 10, sizeof(cl_mem), &amem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 11, sizeof(cl_mem), &dmem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 12, sizeof(int),    &layers);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}

		cout<<"\n\nKernelArgs:";
		//cout<<"\npbuf="<<pbuf;			//0  "p" Projection matrix, holds an array of 12 images, each as as 2D float array.
		cout<<"\n\nprojection matrix, as floats:\n";
		for(int p_iter=0;p_iter<12;p_iter++){cout<<"p["<<p_iter<<"]="<<p[p_iter]<<"\t\t"; if((p_iter+1)%4==0){cout<<"\n";};}
		cout<<"\n";
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

		cout << "\n\ncalcCostVol chk7," << flush;
		cl_event ev;
		cout<<"\n\nclEnqueueNDRangeKernel(";
		cout<<"\ncommand_queue="<<m_queue;
		cout<<"\nkernel=cost_kernel="<<cost_kernel;
		cout<<"\nwork_dim="<<1;
		cout<<"\nglobal_work_offset="<<"NULL"/*0*/;
		cout<<"\nglobal_work_size="<<global_work_size;
		cout<<"\nlocal_work_size="<<local_work_size/*"NULL"*//*0*/;
		cout<<"\nnum_events_in_wait_list"<<0;
		cout<<"\nevent_wait_list=NULL";
		cout<<"\nevent="<<ev;
		cout<<"\n)"<<flush;

		res = clEnqueueNDRangeKernel(m_queue, cost_kernel, 1, NULL/*0*/, &global_work_size, &local_work_size /*NULL*//*0*/, 0, NULL, &ev); /// call cost_kernel ############
		if (res != CL_SUCCESS)	{ cout << "\nclEnqueueNDRangeKernel res = " << checkerror(res) <<"\n"<<flush; exit_(res);}

		status = clFinish(m_queue);			// clFinish needs command_queue, not event list.
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
			width * height * pixelSize,
			image.data,
			0,
			NULL,
			&writeEvt);
		if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer imgmem status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
		cout << "\nimgmem = "<< imgmem <<flush;
		cout << "\nimgmem image_size_bytes= "<< image_size_bytes <<flush;
		cout << "\nwidth * height * pixelSize = " << width * height * pixelSize <<flush;
		cout << "\nimage.size=="<<image.size<<", width="<<width<<", height="<<height <<flush;

		cv::Mat temp_mat(baseImage.size(), baseImage.type());
		ReadOutput(temp_mat.data, imgmem,  image_size_bytes);
		cv::imshow("imgmem_downloaded", temp_mat);
		//cv::waitKey(-1);
																			// flush command queue to device 			// wait for device to complete execution
		status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) clEnqueueWriteBuffer imgmem status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
		status = clWaitForEvents(1, &writeEvt); if (status != CL_SUCCESS)	{ cout << "\nclWaitForEvents clEnqueueWriteBuffer imgmem status = "<<status<<"  "<<checkerror(status) <<"\n"<<flush; exit_(status); }

		cout << "\n\nm_queue=" <<m_queue<<
				"\n pbuf=" <<pbuf<<
				"\n CL_FALSE=" <<CL_FALSE<<
				"\n p=" <<p<<
				"\n &writeEvt=" <<&writeEvt<< flush;

		status = clEnqueueWriteBuffer(m_queue,
			pbuf,
			CL_FALSE,
			0,
			12 * sizeof(float),
			p,
			0,
			NULL,
			&writeEvt);
		if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer pbuf status = " << checkerror(status) <<"\n"<<flush; exit_(status);}

		cout << "\n\ncalcCostVol chk9, m_queue="<<m_queue<< flush;			// flush command queue to device			// clFinish needs command_queue, not event list.
		status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
		status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}

		int layerstep = width * height;

		cout << "\ncalcCostVol chk10," << flush;	// set kernelArg
		res = clSetKernelArg(cost_kernel, 0, sizeof(cl_mem), &pbuf);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 2, sizeof(cl_mem), &imgmem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 5, sizeof(int),    &layerstep);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 6, sizeof(float),  &thresh);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 7, sizeof(int),    &width);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		res = clSetKernelArg(cost_kernel, 12, sizeof(int),   &layers);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}

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

		res = clEnqueueNDRangeKernel(m_queue, cost_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &ev);
		if (res != CL_SUCCESS)	{ cout << "\nclEnqueueNDRangeKernel res = " << checkerror(res) <<"\n"<<flush; exit_(res);}

	} // End of long "if(first image){..}else{..}"///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/// Debug chk
	cv::Mat temp_mat(baseImage.size(), baseImage.type());
	ReadOutput(temp_mat.data, basemem,  image_size_bytes);
	cv::imshow("Img_downloaded_after_cost_kernel", temp_mat);
	cv::waitKey(100);

	cout << "\ncalcCostVol chk13_finished\n" << flush;
/*
	// Enqueue read output buffer
	//cl_event readEvt;
	//status = clEnqueueReadBuffer(m_queue,
	//							cdatabuf,
	//							CL_FALSE,
	//							0,
	//							width * height * layers * sizeof(float),
	//							cdata,
	//							0,
	//							NULL,
	//							&readEvt);

	//status = clEnqueueReadBuffer(m_queue,
	//							hdatabuf,
	//							CL_FALSE,
	//							0,
	//							width * height * layers * sizeof(float),
	//							hdata,
	//							0,
	//							NULL,
	//							&readEvt);
//	status = clFlush(m_queue);
//	status = waitForEventAndRelease(&readEvt);
*/
}// End of RunCL::calcCostVol(...) ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
//void RunCL::minv(float *loInd, float *loVal, int layers)
//{
//	cl_int status;
//	cl_int res;
//	size_t si = height * width * sizeof(float);
//
//	cl_mem loi = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, si, 0, &res);
//	cl_mem lov = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, si, 0, &res);
//
//	if (0 == loi || CL_SUCCESS != res || 0 == lov)
//		return;
//		
//	int layerstep = width * height;
//	// set kernelArg
//	res = clSetKernelArg(min_kernel, 0, sizeof(cl_mem), &cdatabuf);
//	res = clSetKernelArg(min_kernel, 1, sizeof(cl_mem), &loi);
//	res = clSetKernelArg(min_kernel, 2, sizeof(cl_mem), &lov);
//	res = clSetKernelArg(min_kernel, 3, sizeof(int), &height);
//	res = clSetKernelArg(min_kernel, 4, sizeof(int), &layerstep);
//	cl_event ev;
//	res = clEnqueueNDRangeKernel(m_queue, min_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);
//	
//	// Enqueue read output buffer
//	cl_event readEvt;
//	status = clEnqueueReadBuffer(m_queue,
//		loi,
//		CL_FALSE,
//		0,
//		si,
//		loInd,
//		0,
//		NULL,
//		&readEvt);
//	
//	status = clFlush(m_queue);
//	status = waitForEventAndRelease(&readEvt);
//}
 
//void RunCL::optiQ(float sigma_q,float epsilon,float denom)
//{
//	cl_event ev;
//		
//	cl_int res;
//	res = clSetKernelArg(optiQ_kernel, 0, sizeof(float), &sigma_q);
//	res = clSetKernelArg(optiQ_kernel, 1, sizeof(float), &epsilon);
//	res = clSetKernelArg(optiQ_kernel, 2, sizeof(float), &denom);
//	res = clSetKernelArg(optiQ_kernel, 3, sizeof(int), &width);
//	res = clSetKernelArg(optiD_kernel, 4, sizeof(int), &height);
//	res = clSetKernelArg(optiQ_kernel, 5, sizeof(cl_mem), &qxmem);
//	res = clSetKernelArg(optiQ_kernel, 6, sizeof(cl_mem), &qymem);
//	res = clSetKernelArg(optiQ_kernel, 7, sizeof(cl_mem), &dmem);
//	res = clSetKernelArg(optiQ_kernel, 8, sizeof(cl_mem), &grmem);
//	res = clSetKernelArg(optiQ_kernel, 9, sizeof(cl_mem), &gdmem);
//	
//	res = clEnqueueNDRangeKernel(m_queue, optiQ_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);
//}

//void RunCL::optiD(float sigma_d, float epsilon, float denom, float theta)
//{
//	// set kernelArg
//	cl_event ev;
//	cl_int res = clSetKernelArg(optiD_kernel, 0, sizeof(float), &sigma_d);
//	res = clSetKernelArg(optiD_kernel, 1, sizeof(float), &epsilon);
//	res = clSetKernelArg(optiD_kernel, 2, sizeof(float), &denom);
//	res = clSetKernelArg(optiD_kernel, 3, sizeof(int), &width);
//	res = clSetKernelArg(optiD_kernel, 4, sizeof(int), &height);
//	res = clSetKernelArg(optiD_kernel, 5, sizeof(cl_mem), &qxmem);
//	res = clSetKernelArg(optiD_kernel, 6, sizeof(cl_mem), &qymem);
//	res = clSetKernelArg(optiD_kernel, 7, sizeof(cl_mem), &dmem);
//	res = clSetKernelArg(optiD_kernel, 8, sizeof(cl_mem), &gumem);
//	res = clSetKernelArg(optiD_kernel, 9, sizeof(cl_mem), &glmem);
//	res = clSetKernelArg(optiD_kernel, 10, sizeof(cl_mem), &grmem);
//	res = clSetKernelArg(optiD_kernel, 11, sizeof(cl_mem), &gdmem); 
//	res = clSetKernelArg(optiD_kernel, 12, sizeof(cl_mem), &amem);
//	res = clSetKernelArg(optiD_kernel, 13, sizeof(float), &theta);
//	
//	res = clEnqueueNDRangeKernel(m_queue, optiD_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);
//}

//void RunCL::optiA(float theta,float ds,float lamda,int l,int layerstep)
//{
//	cl_event ev;
//	cl_int res;
//	res = clSetKernelArg(optiA_kernel, 0, sizeof(cl_mem), &amem);
//	res = clSetKernelArg(optiA_kernel, 1, sizeof(cl_mem), &dmem);
//	res = clSetKernelArg(optiA_kernel, 2, sizeof(cl_mem), &cdatabuf);
//	res = clSetKernelArg(optiA_kernel, 3, sizeof(float), &theta);
//	res = clSetKernelArg(optiA_kernel, 4, sizeof(float), &ds);
//	res = clSetKernelArg(optiA_kernel, 5, sizeof(float), &lamda);
//	res = clSetKernelArg(optiA_kernel, 6, sizeof(int), &l);
//	res = clSetKernelArg(optiA_kernel, 7, sizeof(int), &layerstep);
//	res = clSetKernelArg(optiA_kernel, 8, sizeof(int), &width);
//
//	res = clEnqueueNDRangeKernel(m_queue, optiA_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);
//}
*/

void RunCL::allocatemem(float *qx,float *qy, float* gx, float* gy)
{
	cout << "RunCL::allocatemem_chk0" << flush;
	cl_int res;    // | (1 << 6)
	qxmem  = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res); 	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	qymem  = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	dmem   = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	amem   = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	gxmem  = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	gymem  = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	gqxmem = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	gqymem = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	g1mem  = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	lomem  = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	himem  = clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	cout << "qxmem = " << qxmem << endl;
	cout << "qymem = " << qymem << endl;
	cout << "dmem = " << dmem << endl;
	cout << "amem = " << amem << endl;
	cout << "gxmem = " << gxmem << endl;
	cout << "gymem = " << gymem << endl;
	cout << "gqxmem = " << gqxmem << endl;
	cout << "gqymem = " << gqymem << endl;
	cout << "g1mem = " << g1mem << endl;
	cout << "lomem = " << lomem << endl;
	cout << "himem = " << himem << endl;
	cout << " " << endl;

	cout << "RunCL::allocatemem_chk1\n" << flush;
	cl_int status;
	cl_event writeEvt;
	status = clEnqueueWriteBuffer(m_queue, // WriteBuffer qxmem ##########
		qxmem,
		CL_FALSE,
		0,
		width * height,
		qx,
		0,
		NULL,
		&writeEvt);

	if (status != CL_SUCCESS)	{
		cout << "Error: allocatemem_chk1.1" << endl;
		cout << "status = " << status << " " << checkerror(status) << endl;
		cout << "m_queue = " << m_queue << endl;
		cout << "qxmem = " << qxmem << endl;
		cout << "CL_FALSE = " << CL_FALSE << endl;
		cout << "width * height = " << width * height << endl;
		cout << "qx = " << qx << endl;
		cout << "&writeEvt = " << &writeEvt  << endl;
		cout << " \n" << flush;
		exit_(status);//return;
	}
		else{cout << "allocatemem_chk1.1\n" << flush;}
    
	status = clEnqueueWriteBuffer(m_queue, // WriteBuffer qymem ##########
		qymem,
		CL_FALSE,
		0,
		width * height,
		qy,
		0,
		NULL,
		&writeEvt);
	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.2\n" << endl; exit_(status);}
	else{cout << "allocatemem_chk1.2\n" << flush;}
	
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
	cout << "RunCL::allocatemem_finished\n\n" << flush;
}// End of RunCL::allocatemem(..) ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RunCL::~RunCL() // destructor
{
	cl_int status;
	status = clReleaseKernel(cost_kernel);
	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
    /*
	//status = clReleaseKernel(min_kernel);
	//status = clReleaseKernel(optiQ_kernel);
	//status = clReleaseKernel(optiD_kernel);
	//status = clReleaseKernel(optiA_kernel);
    */
	status = clReleaseKernel(cache1_kernel);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseKernel(cache2_kernel);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseKernel(updateQ_kernel);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseKernel(updateD_kernel);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
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
	status = clReleaseMemObject(pbuf);		if (status != CL_SUCCESS)	{ cout << "\npbuf     status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.5"<<flush;
	status = clReleaseMemObject(qxmem);		if (status != CL_SUCCESS)	{ cout << "\nqxmem    status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.6"<<flush;
	status = clReleaseMemObject(qymem);		if (status != CL_SUCCESS)	{ cout << "\nqymem    status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.7"<<flush;
	status = clReleaseMemObject(gqxmem);	if (status != CL_SUCCESS)	{ cout << "\ngqxmem   status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.8"<<flush;
	status = clReleaseMemObject(gqymem);	if (status != CL_SUCCESS)	{ cout << "\ngqymem   status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.9"<<flush;
	status = clReleaseMemObject(dmem);		if (status != CL_SUCCESS)	{ cout << "\ndmem     status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.10"<<flush;
	status = clReleaseMemObject(amem);		if (status != CL_SUCCESS)	{ cout << "\namem     status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.11"<<flush;
	status = clReleaseMemObject(lomem);		if (status != CL_SUCCESS)	{ cout << "\nlomem    status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.12"<<flush;
	status = clReleaseMemObject(himem);		if (status != CL_SUCCESS)	{ cout << "\nhimem    status = " << checkerror(status) <<"\n"<<flush; }		cout<<"\nRunCL::CleanUp_chk0.13"<<flush;
	/*
    status = clReleaseMemObject(gdmem);
	status = clReleaseMemObject(gumem);
	status = clReleaseMemObject(glmem);
	status = clReleaseMemObject(grmem);
	*/
	cout<<"\nRunCL::CleanUp_chk1_finished"<<flush;
}

void RunCL::exit_(cl_int res)
{
	CleanUp();
	//~RunCL(); Never need to call a destructor manually.
	exit(res);
}

void RunCL::cacheGValue(cv::Mat &bgray)
{
	cout<<"\ncacheGValue_chk0"<<flush;
	cl_int status;
	cl_int res;
	size_t s = bgray.total();
	cl_event ev;

	if (basegraymem == 0) {
		cout<<"\ncacheGValue_chk1"<<flush;
		basegraymem = clCreateBuffer(m_context, CL_MEM_READ_ONLY , s, 0, &res); // CL_MEM_READ_ONLY | (1 << 6)

		cout<<"\ncacheGValue_chk2"<<flush;
		status = clEnqueueWriteBuffer(m_queue, // WriteBuffer basegraymem ##########
			basegraymem,
			CL_FALSE,
			0,
			width * height,
			bgray.data,
			0,
			NULL,
			&ev);
		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}  // CL_INVALID_MEM_OBJECT // CL_OUT_OF_RESOURCES ///////////////
	}
	cout<<"\ncacheGValue_chk3"<<flush;

	res = clSetKernelArg(cache1_kernel, 0, sizeof(cl_mem), &basegraymem);if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cache1_kernel, 1, sizeof(cl_mem), &g1mem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cache1_kernel, 2, sizeof(int),    &width);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cache1_kernel, 3, sizeof(int),    &height);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	cout<<"\ncacheGValue_chk4"<<flush;
	res = clEnqueueNDRangeKernel(m_queue, cache1_kernel, 1, NULL/*0*/, &global_work_size, &local_work_size, 0, NULL, &ev); // run cache1_kernel  aka CacheG1(..) ################
	if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}

	cout<<"\ncacheGValue_chk5"<<flush;
	/*
    size_t st = width * height * sizeof(float);
	float* g1 = (float*)malloc(st);
	cl_event readEvt;
	status = clEnqueueReadBuffer(m_queue,  // ReadBuffer g1mem ###########
		g1mem,
		CL_FALSE,
		0,
		st,
		g1,
		0,
		NULL,
		&readEvt);
	status = clFlush(m_queue);
	status = waitForEventAndRelease(&readEvt);
	cv::Mat qx;
	qx.create(height, width, CV_32FC1); qx.reshape(0, height);
	memcpy(qx.data, g1, st);
	double min = 0, max = 0;
	cv::minMaxIdx(qx, &min, &max);
	*/
	res = clSetKernelArg(cache2_kernel, 0, sizeof(cl_mem), &g1mem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cache2_kernel, 1, sizeof(cl_mem), &gxmem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cache2_kernel, 2, sizeof(cl_mem), &gymem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cache2_kernel, 3, sizeof(int),    &width);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cache2_kernel, 4, sizeof(int),    &height);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	cout<<"\ncacheGValue_chk6"<<flush;
	res = clEnqueueNDRangeKernel(m_queue, cache2_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &ev); // run cache2_kernel  aka CacheG2(..) #####################
	if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}

	cout<<"\ncacheGValue_chk7"<<flush;
	// download & save buffers that were writtent to.







	cout<<"\ncacheGValue_chk_8_finished"<<flush;
}


void RunCL::updateQD(float epsilon, float theta, float sigma_q, float sigma_d)
{
	cout<<"\nupdateQD_chk0"<<flush;
	cl_int status;
	cl_int res;
	cl_event ev; 
	
	res = clSetKernelArg(updateQ_kernel, 0, sizeof(cl_mem), &gxmem);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateQ_kernel, 1, sizeof(cl_mem), &gymem);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateQ_kernel, 2, sizeof(cl_mem), &gqxmem);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateQ_kernel, 3, sizeof(cl_mem), &gqymem);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateQ_kernel, 4, sizeof(cl_mem), &dmem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateQ_kernel, 5, sizeof(float),  &epsilon);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateQ_kernel, 6, sizeof(float),  &theta);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateQ_kernel, 7, sizeof(int),    &width);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateQ_kernel, 8, sizeof(int),    &height);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	
	cout<<"\nupdateQD_chk1"<<flush;
	res = clEnqueueNDRangeKernel(m_queue, updateQ_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &ev); // run updateQ_kernel  aka UpdateQ(..) ####################
	if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}

	cout<<"\nupdateQD_chk2"<<flush;
	size_t st = width * height * sizeof(float);
	float* gqx = (float*)malloc(st);
	cl_event readEvt;

	cout<<"\nupdateQD_chk3"<<flush;
	status = clEnqueueReadBuffer(m_queue,  // readBuffer  gqxmem #########
		gqxmem,
		CL_FALSE,
		0,
		st,
		gqx,
		0,
		NULL,
		&readEvt);
	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}

	cout<<"\nupdateQD_chk4"<<flush;
	status = clFlush(m_queue);					if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = waitForEventAndRelease(&readEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}

	cout<<"\nupdateQD_chk5"<<flush;
	cv::Mat qx;
	qx.create(1, width*height, CV_32FC1);
	memcpy(qx.data, gqx, st);
	double min = 0, max = 0;
	cv::minMaxIdx(qx, &min, &max);

	cout<<"\nupdateQD_chk6"<<flush;
	res = clSetKernelArg(updateD_kernel, 0, sizeof(cl_mem), &gqxmem);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateD_kernel, 1, sizeof(cl_mem), &gqymem);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateD_kernel, 2, sizeof(cl_mem), &dmem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateD_kernel, 3, sizeof(cl_mem), &amem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateD_kernel, 4, sizeof(float),  &theta);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateD_kernel, 5, sizeof(float),  &sigma_d);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateD_kernel, 6, sizeof(int),    &width);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	cout<<"\nupdateQD_chk7"<<flush;
	res = clEnqueueNDRangeKernel(m_queue, updateD_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &ev); // run updateD_kernel  aka UpdateD(..) ####################
	if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}


	cout<<"\nupdateQD_chk8"<<flush;
	// download & save buffers that were writtent to.







	cout<<"\nupdateQD_chk9_finished\n"<<flush;
}


void RunCL::updateA(int layers, float lambda,float theta)
{
	cl_int status;
	cl_int res;
	cl_event ev;

	res = clSetKernelArg(updateA_kernel, 0, sizeof(cl_mem), &cdatabuf); if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateA_kernel, 1, sizeof(cl_mem), &amem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateA_kernel, 2, sizeof(cl_mem), &dmem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateA_kernel, 3, sizeof(int),    &layers);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateA_kernel, 4, sizeof(int),    &width);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateA_kernel, 5, sizeof(int),    &height);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateA_kernel, 6, sizeof(float),  &lambda);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(updateA_kernel, 7, sizeof(float),  &theta);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	res = clEnqueueNDRangeKernel(m_queue, updateA_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &ev); // run updateA_kernel  aka UpdateA(..) ####################
	if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}

	cout<<"\nupdateA_chk1"<<flush;
	// download & save buffers that were writtent to.










	cout<<"\nupdateA_chk2_finished"<<flush;
}
