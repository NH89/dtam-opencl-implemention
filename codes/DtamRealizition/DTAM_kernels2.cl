/*
// the data in float p[] , set in RunCL.cpp,    void RunCL::calcCostVol(...),   variables declared in RunCL.h  class RunCL{...}
//  res = clSetKernelArg(cost_kernel, 0, sizeof(cl_mem),    &pbuf);
//  res = clSetKernelArg(cost_kernel, 1, sizeof(cl_mem),    &basemem);
//  res = clSetKernelArg(cost_kernel, 2, sizeof(cl_mem),    &imgmem);
//  res = clSetKernelArg(cost_kernel, 3, sizeof(cl_mem),    &cdatabuf);
//  res = clSetKernelArg(cost_kernel, 4, sizeof(cl_mem),    &hdatabuf);
//  res = clSetKernelArg(cost_kernel, 5, sizeof(int),       &layerstep);
//  res = clSetKernelArg(cost_kernel, 6, sizeof(float),     &thresh);
//  res = clSetKernelArg(cost_kernel, 7, sizeof(int),       &width);
//  res = clSetKernelArg(cost_kernel, 8, sizeof(cl_mem),    &lomem);
//  res = clSetKernelArg(cost_kernel, 9, sizeof(cl_mem),    &himem);
//  res = clSetKernelArg(cost_kernel, 10, sizeof(cl_mem),   &amem);
//  res = clSetKernelArg(cost_kernel, 11, sizeof(cl_mem),   &dmem);
//  res = clSetKernelArg(cost_kernel, 12, sizeof(int),      &layers);
*/

#define pixels_			0  // Can these be #included from a common header for both host and device code?
#define rows_			1
#define cols_			2
#define layers_			3
#define max_inv_depth_	4
#define min_inv_depth_	5
#define inv_d_step_		6
#define alpha_g_		7
#define beta_g_			8	///  __kernel void CacheG4
#define epsilon_		9	///  __kernel void UpdateQD		// epsilon = 0.1
#define sigma_q_		10									// sigma_q = 0.0559017
#define sigma_d_		11
#define theta_			12
#define lambda_			13	///   __kernel void UpdateA2
#define scale_Eaux_		14
__kernel void BuildCostVolume2(						// called as "cost_kernel" in RunCL.cpp
// TODO rewrite with homogeneuos coords to handle points at infinity (x,y,z,0) -> (u,v,0)
	__global float* k2k,		//0
	__global float* base,		//1  // uchar*
	__global float* img,		//2  // uchar*
	__global float* cdata,		//3
	__global float* hdata,		//4  'w' num times cost vol elem has been updated
	__global float* lo, 		//5
	__global float* hi,			//6
	__global float* a,			//7
	__global float* d,			//8
	__constant float* params)	//9 pixels, rows, cols, layers, max_inv_depth, min_inv_depth, inv_d_step, threshold
{
	int global_id = get_global_id(0);

	int pixels 			= floor(params[pixels_]);
	int rows 			= floor(params[rows_]);
	int cols 			= floor(params[cols_]);
	int layers			= floor(params[layers_]);
	float max_inv_depth = params[max_inv_depth_];  // not used
	float min_inv_depth = params[min_inv_depth_];
	float inv_d_step 	= params[inv_d_step_];

	// keyframe pixel coords
	float u = global_id % cols;
	float v = (int)(global_id / cols);

	// Get keyframe pixel values
	int offset_3 = global_id *3;
	float3 B;	B.x = base[offset_3]; B.y = base[offset_3]; B.z = base[offset_3];	// pixel from keyframe

	// variables for the cost vol
	float u2,v2, inv_depth=0.0;
	int int_u2, int_v2, cv_idx = global_id;
	int coff_00, coff_01, coff_10, coff_11;
	float3 c, c_00, c_01, c_10, c_11;
	float rho;
	float ns=0.0, mini=0.0, minv=3.0, maxv=0.0; // 3.0 would be max rho for 3 chanels.
	float c0 = cdata[cv_idx];	// cost for this elem of cost vol
	float w  = hdata[cv_idx];	// count of updates of this costvol element. w = 001 initially
	int layer = 0;
	// layer zero, ////////////////////////////////////////////////////////////////////////////////////////
	// inf depth, roation without paralax, i.e. reproj without translation.
	// Use depth=1 unit sphere, with rotational-preprojection matrix

	// precalculate depth-independent part of reprojection, h=homogeneous coords.
	float uh2 = k2k[0]*u + k2k[1]*v + k2k[2]*1;  // +k2k[3]/z
	float vh2 = k2k[4]*u + k2k[5]*v + k2k[6]*1;  // +k2k[7]/z
	float wh2 = k2k[8]*u + k2k[9]*v + k2k[10]*1; // +k2k[11]/z
	//float h/z  = k2k[12]*u + k2k[13]*v + k2k[14]*1; // +k2k[15]/z

	// cost volume loop  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	#define MAX_LAYERS 64
	float cost[MAX_LAYERS];
	for( layer=0;  layer<layers; layer++ ){
		cv_idx = global_id + layer*pixels;
		/*c0*/ cost[layer] = cdata[cv_idx];	// cost for this elem of cost vol
		w  = hdata[cv_idx];	// count of updates of this costvol element. w = 001 initially

		// locate pixel to sample from  new image. Depth dependent part.
		inv_depth = (layer * inv_d_step) + min_inv_depth;
		uh2  = uh2 + k2k[3]*inv_depth;
		vh2  = vh2 + k2k[7]*inv_depth;
		wh2  = wh2 + k2k[11]*inv_depth;
		u2   = uh2/wh2;
		v2   = vh2/wh2;

		int_u2 = ceil(u2);
		int_v2 = ceil(v2);

		if ( !((int_u2<1) || (int_u2>cols-2) || (int_v2<1) || (int_v2>rows-2)) ) {  	// if (not within new frame) skip
			// compute adjacent pixel indices
			coff_11 =  int_v2    * cols +  int_u2;
			coff_10 = (int_v2-1) * cols +  int_u2;
			coff_01 =  int_v2    * cols + (int_u2 -1);
			coff_00 = (int_v2-1) * cols + (int_u2 -1);

			// sample adjacent pixels
			c_11.x= img[coff_11*3]; c_11.y=img[1+(coff_11*3)], c_11.z=img[2+(coff_11*3)];		// c.xyz = rgb values of pixel in new frame
			c_10.x= img[coff_10*3]; c_10.y=img[1+(coff_10*3)], c_10.z=img[2+(coff_10*3)];
			c_01.x= img[coff_01*3]; c_01.y=img[1+(coff_01*3)], c_01.z=img[2+(coff_01*3)];
			c_00.x= img[coff_00*3]; c_00.y=img[1+(coff_00*3)], c_00.z=img[2+(coff_00*3)];

			// weighting for bi-linear interpolation
			float factor_x = fmod(u2,1);
			float factor_y = fmod(v2,1);
			c = factor_y * (c_11*factor_x  + c_01*(1-factor_x)) + (1-factor_y) * (c_10*factor_x  + c_00*(1-factor_x));  // float3, bi-linear interpolation

			// Compute photometric cost
			rho = ( fabs(c.x-B.x) + fabs(c.y-B.y) + fabs(c.z-B.z) )/(3.0);	//					// L1 norm between keyframe & new frame pixels.
			cost[layer] = (cost[layer]*w + rho) / (w + 1);				// cost[layer] = existing value in this costvol elem.
			cdata[cv_idx] = cost[layer];  //rho;//c.x + c.y + c.z; // Costdata, same location cost[layer] was read from.  // CostVol set here ###########
			hdata[cv_idx] = w + 1;		//B.x + B.y+ B.z; //c.x + c.y + c.z; //rho; //			// Weightdata, counts updates of this costvol element.
		}
	}
	for( layer=0;  layer<layers; layer++ ){
		if (cost[layer] < minv) { 		// find idx & value of min cost in this ray of cost vol, given this update. NB Use array private to this thread.
			minv = cost[layer];
			mini = layer;
		}
		maxv = fmax(cost[layer], maxv);
	}

	// Print out data at one pixel, at each layer:
	if(u==70 && v==470){
		// (u==407 && v==60) corner of picture,  (u==300 && v==121) centre of monitor, (u==171 && v==302) on the desk, (470, 70) out of frame in later images
		float d;
		if (inv_depth==0){d=0;}
		else {d=1/inv_depth;}
		printf("\nlayer=%i, inv_depth=%f, depth=%f, um=%f, vm=%f, rho=%f, trans=(%f,%f,%f), c0=%f, w=%f, ns=%f, rho=%f, minv=%f, mini=%f, params[%i,%i,%i,%i,%f,%f,%f,]", \
		layer, inv_depth, d, u2, v2, rho, k2k[3], k2k[7], k2k[11], c0, w, ns, rho, minv, mini,\
			pixels, rows, cols, layers, max_inv_depth, min_inv_depth, inv_d_step
		);
	}

	lo[global_id] 	= minv; 			// min photometric cost  // rho;//
	a[global_id] 	= mini*inv_d_step + min_inv_depth;// c.x + c.y + c.z; //mini*inv_d_step; 	// inverse distance      //  int_u2;//
	d[global_id] 	= mini*inv_d_step + min_inv_depth;//ns;   	// photometric cost in the last layer of cost vol. // not a good initiallization of d[] ?   // int_v2;//
	hi[global_id] 	= maxv; 			// max photometric cost

	// refinement step - from DTAM Mapping

	if(mini == 0 || mini == layers-1) return;// first or last inverse depth was best
	const float A_ = cdata[(int)(global_id + (mini-1)*pixels)]  ;//Cost[i+(minl-1)*layerStep];
	const float B_ = minv;
	const float C_ = cdata[(int)(global_id + (mini+1)*pixels)]  ;//Cost[i+(minl+1)*layerStep];
	float delta = ((A_+C_)==2*B_)? 0.0f : ((C_-A_)*inv_d_step)/(2*(A_-2*B_+C_));
	delta = (fabs(delta) > inv_d_step)? 0.0f : delta;// if the gradient descent step is less than one whole inverse depth interval, reject interpolation
	a[global_id] += delta;																		// CminIdx[i] = .. ////
	if ((global_id%1000 ==0) &&  (delta != 0.0)) printf("(%f,%f,%f),", mini*inv_d_step, delta, inv_d_step);		// NB CminIdx used to initialize A & D
	// NB at present delta is too small!
}


 __kernel void CacheG3(
	 __global float* base,
	 __global float* gxp,
	 __global float* gyp,
	 __global float* g1p,
	 __constant float* params
 )
 {
	 int x = get_global_id(0);
	 int rows 			= floor(params[rows_]);
	 int cols 			= floor(params[cols_]);
	 float alphaG		= params[alpha_g_];
	 float betaG 		= params[beta_g_];

	 int y = x / cols;
	 x = x % cols;
	 if (x<2 || x > cols-2 || y<2 || y>rows-2) return;  // needed for wider kernel
	 int upoff = -(y != 0)*cols;                   // up, down, left, right offsets, by boolean logic.
	 int dnoff = (y < rows-1) * cols;
     int lfoff = -(x != 0);
	 int rtoff = (x < cols-1);

	 barrier(CLK_GLOBAL_MEM_FENCE);

	 unsigned int offset = x + y * cols;

	 float pu, pd, pl, pr;                         // rho, photometric difference: up, down, left, right, of grayscale ref image.
	 float g0x, g0y, g0, g1;

	 pr =  base[offset + rtoff];// + base[offset + rtoff +1];					   // NB base = grayscale CV_8UC1 image.
	 pl =  base[offset + lfoff];// + base[offset + lfoff -1];
	 pu =  base[offset + upoff];// + base[offset + 2*upoff];
	 pd =  base[offset + dnoff];// + base[offset + 2*dnoff];

	 float gx, gy;
	 gx			= fabs(pr - pl);
	 gy			= fabs(pd - pu);
	 g1p[offset]= exp(-alphaG * pow(sqrt(gx*gx + gy*gy), betaG) );
	 gxp[offset]= gx;
	 gyp[offset]= gy;

	 if (x==100 && y==100) printf("\n\nCacheG4, params[alpha_g_]=%f, params[beta_g_]=%f, gxp[offset]=%f, gyp[offset]=%f, g1p[offset]=%f, sqrt(gx*gx + gy*gy)=%f, pow(sqrt(gx*gx + gy*gy), betaG)=%f \n", params[alpha_g_], params[beta_g_], gxp[offset], gyp[offset], g1p[offset], sqrt(gx*gx + gy*gy), pow(sqrt(gx*gx + gy*gy), betaG) );
 }


 __kernel void UpdateQD(
	 __global float* g1pt,
	 __global float* qpt,
	 __global float* dpt,                           // dmem,     depth D
	 __global float* apt,                           // amem,     auxilliary A
	 __constant float* params
	 )
 {
	 int g_id = get_global_id(0);
	 int rows 			= floor(params[rows_]);
	 int cols 			= floor(params[cols_]);
	 float epsilon 		= params[epsilon_];
	 float sigma_q 		= params[sigma_q_];
	 float sigma_d 		= params[sigma_d_];
	 float theta 		= params[theta_];

	 int y = g_id / cols;
	 int x = g_id % cols;
	 unsigned int pt = x + y * cols;               // index of this pixel
	 barrier(CLK_GLOBAL_MEM_FENCE);
	 const int wh = (cols*rows);

	 float g1 = g1pt[pt];
	 float qx = qpt[pt];
	 float qy = qpt[pt+wh];
	 float d  = dpt[pt];
	 float a  = apt[pt];

	 const float dd_x = (x==cols-1)? 0.0f : dpt[pt+1]    - d;			// Sample depth gradient in x&y
	 const float dd_y = (y==rows-1)? 0.0f : dpt[pt+cols] - d;
	 // DTAM paper, primal-dual update step

	 qx = (qx + sigma_q*g1*dd_x) / (1.0f + sigma_q*epsilon);
	 qy = (qy + sigma_q*g1*dd_y) / (1.0f + sigma_q*epsilon);
	 const float maxq = fmax(1.0f, sqrt(qx*qx + qy*qy));

	 qx = qx/maxq;
	 qy = qy/maxq;

	 qpt[pt]    = qx;  //dd_x;//pt2;//wh;//pt;//dd_x;//qx / maxq;
	 qpt[pt+wh] = qy;  //dd_y;//pt;//;//y;//dd_y;//dpt[pt+1] - d; //dd_y;//qy / maxq;

	 barrier(CLK_GLOBAL_MEM_FENCE);						// needs to be after all Q updates. TODO ? is this fence sufficient ?
	 float dqx_x;// = div_q_x(q, w, x, i);											// div_q_x(..)
	 if (x == 0) dqx_x = qx;
	 else if (x == cols-1) dqx_x =  -qpt[pt-1];
	 else dqx_x =  qx- qpt[pt-1];

	 float dqy_y;// = div_q_y(q, w, h, wh, y, i);										// div_q_y(..)
	 if (y == 0) dqy_y =  qy;							// return q[i];
	 else if (y == rows-1) dqy_y = -qpt[pt+wh-cols];	// return -q[i-1];
	 else dqy_y =  qy - qpt[pt+wh-cols];				// return q[i]- q[i-w];

	 const float div_q = dqx_x + dqy_y;

	 dpt[pt] = (d + sigma_d*(g1*div_q + a/theta)) / (1.0f + sigma_d/theta);

	 if (x==100 && y==100) printf("\ndpt[pt]=%f, d=%f, sigma_d=%f, g1=%f, div_q=%f, a=%f, theta=%f, sigma_d=%f, theta=%f : qx=%f, qy=%f, maxq=%f, dd_x=%f, dd_y=%f ", dpt[pt], d, sigma_d, g1, div_q , a, theta, sigma_d, theta, qx, qy, maxq, dd_x, dd_y );
 }

// 					( inverse_depth, r , min_inv_depth, inv_depth_step, num_layers )
int set_start_layer(float di, float r, float far, float depthStep, int layers, int x, int y){
    const float d_start = di - r;
    const int start_layer =  floor( (d_start - far)/depthStep )    ;  //floor((d_start - far)/depthStep) - 1;
	//if (x==200 && y==200) printf("\n# start_layer=%i, d_start=%f, di=%f, r=%f, far=%f, depthStep=%f", start_layer, d_start, di, r, far, depthStep );
    return (start_layer<0)? 0 : start_layer;		// start_layer >= 0
}

int set_end_layer(float di, float r, float far, float depthStep, int layers, int x, int y){
    const float d_end = di + r;
    const int end_layer = ceil((d_end - far)/depthStep) + 1;        // lrintf = Round input to nearest integer value
    // int end_layer = 255;// int layer = int((d_end - far)/depthStep) + 1;
    //if (x==200 && y==200) printf("\n# end_layer=%i", end_layer );
    return  (end_layer>(layers-1))? (layers-1) : end_layer;			// end_layer <= layers-1
}

float get_Eaux(float theta, float di, float aIdx, float far, float depthStep, float lambda, float scale_Eaux, float costval)
{
	const float ai = far + aIdx*depthStep;
	return scale_Eaux*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;

	// return (0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;
	// return 100*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;
	// return 10000*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;
	// return 1000000*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;
	// return 10000000*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;
}


 __kernel void UpdateA2(  // pointwise exhaustive search
	__global float* cdata,                         //           cost volume
	__global float* apt,                           // dmem,     depth D
	__global float* dpt,                           // amem,     auxilliary A
	__global float* lo,
	__global float* hi,
	__constant float* params
)
 {
	 int x 					= get_global_id(0);
	 int rows 				= floor(params[rows_]);
	 int cols 				= floor(params[cols_]);
	 int layers				= floor(params[layers_]);
	 unsigned int layer_step = floor(params[pixels_]);
	 float lambda			= params[lambda_];
	 float theta			= params[theta_];
	 float min_d			= params[max_inv_depth_]; //far
	 float max_d			= params[min_inv_depth_]; //near
	 float scale_Eaux		= params[scale_Eaux_];

	 int y = x / cols;
	 x = x % cols;
	 unsigned int pt  = x + y * cols;               // index of this pixel
	 if (pt >= (cols*rows))return;
	 unsigned int cpt = pt;

	 barrier(CLK_GLOBAL_MEM_FENCE);

	 float d  			= dpt[pt];
	 float E_a  		= FLT_MAX;
	 float min_val		= FLT_MAX;
	 int   min_layer	= 0;

	 const float depthStep = params[inv_d_step_]; //(min_d - max_d) / (layers - 1);
	 const int   layerStep = rows*cols;

	 const float r = sqrt( 2*theta*lambda*(hi[pt] - lo[pt]) );
	 const int start_layer = set_start_layer(d, r, min_d, depthStep, layers, x, y);  // 0;//
	 const int end_layer   = set_end_layer(d, r, min_d, depthStep, layers, x, y);  // layers-1; //
	 int minl = 0;
	 float Eaux_min = 1e+30; // set high initial value

	 for(int l = start_layer; l <= end_layer; l++) {
		const float cost_total = get_Eaux(theta, d, (float)l, min_d, depthStep, lambda, scale_Eaux, cdata[pt+l*layerStep]);
		// apt[pt+l*layerStep] = cost_total;  // DTAM_Mapping collects an Eaux volume, for debugging.
		if(cost_total < Eaux_min) {
			Eaux_min = cost_total;
			minl = l;
		}
	 }

	float a = min_d + minl*depthStep;  // NB implicit conversion: int minl -> float.
	/*
	//refinement step
	if(minl > start_layer && minl < end_layer){ //return;// if(minl == 0 || minl == layers-1) // first or last was best
		// sublayer sampling as the minimum of the parabola with the 2 points around (minl, Eaux_min)
		const float A = get_Eaux(theta, d, minl-1, max_d, depthStep, lambda, scale_Eaux, cdata[pt+(minl-1)*layerStep]);
		const float B = Eaux_min;
		const float C = get_Eaux(theta, d, minl+1, max_d, depthStep, lambda, scale_Eaux, cdata[pt+(minl+1)*layerStep]);
		// float delta = ((A+C)==2*B)? 0.0f : ((A-C)*depthStep)/(2*(A-2*B+C));
		float delta = ((A+C)==2*B)? 0.0f : ((C-A)*depthStep)/(2*(A-2*B+C));
		delta = (fabs(delta) > depthStep)? 0.0f : delta;
		// a[i] += delta;
		a -= delta;
	}
	*/
	apt[pt] = a;

	if (x==200 && y==200) printf("\n\nUpdateA: theta=%f, lambda=%f, hi=%f, lo=%f, r=%f, d=%f, min_d=%f, depthStep=%f, layers=%i, start_layer=%i, end_layer=%i", \
		theta, lambda, hi[pt], lo[pt], r, d, min_d, depthStep, layers, start_layer, end_layer );
 }
