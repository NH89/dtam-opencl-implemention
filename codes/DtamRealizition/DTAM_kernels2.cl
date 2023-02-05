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

__kernel void BuildCostVolume(						// called as "cost_kernel" in RunCL.cpp
	__global float* p,		//0  kernelArg numbers
	__global uchar* base,	//1
	__global uchar* img,	//2
	__global float* cdata,	//3
	__global float* hdata,	//4 'w'
	int layerStep,			//5 width*height
	float weight,			//6
	int cols,				//7
	__global float* lo, 	//8
	__global float* hi,		//9
	__global float* a,		//10
	__global float* d,		//11
	int layers)				//12 layers=32
{
	//weight = 0.5;
	int global_id = get_global_id(0);
	int local_id  = get_local_id(0);
	int xf = global_id; 							// TODO either pass rows in kernel args, OR compile const values for rows, cols, layers, layerStep.
	int yf = xf / (cols);
	xf = xf % (cols);

	int rows = layerStep/cols;
	unsigned int offset_1 = xf + yf * cols; 			// i.e offset = global_id
	unsigned int offset_3 = offset_1 * 3;
	float xf_1 = xf;
	float yf_1 = yf;
	float3 B;	B.x = base[offset_3]; B.y = base[offset_3]; B.z = base[offset_3];	// pixel from keyframe
													// (wi,xi,yi)=homogeneous coords of keyframe _ray_. // Below old comments, meaning ??
	float xi = p[0] * xf_1  + p[1] * yf_1  + p[3];	//  xi = p[p_buf]      * global_id       + p[base_mem]   * global_id/cols       + p[c_data_buf];
	float yi = p[4] * xf_1  + p[5] * yf_1  + p[7];	//  yi = p[h_data_buf] * global_id       + p[layer_step] * global_id/cols       + p[width];
	float wi = p[8] * xf_1  + p[9] * yf_1  + p[11];	//  wi = p[lo_mem]     * global_id       + p[hi_mem]     * global_id/cols       + p[d_mem];

	float minv = 1000.0, maxv = 0.0, mini = 0;
	barrier(CLK_GLOBAL_MEM_FENCE);

	unsigned int coff_11, coff_01, coff_10, coff_00;
	float3  c, c_11, c_10, c_01, c_00;
	float v1=0, v2=0, v3=0, del=0, ns=0;

	for (unsigned int z = 0; z < layers/*1*/; z++)  // for layers of cost vol...  TODO prefer inverse depth, better distribution of layer depths.
	{
		float c0 = cdata[offset_1 + z*layerStep];	// cost for this elem of cost vol
		float w  = hdata[offset_1 + z*layerStep];	// weighting for this elem in hdata (weights). w = 001 initially
													// (wiz,xiz,yiz)=homogeneous pixel coords in new frame of keyframe _point_.
													// Below old comments, meaning ??
		float wiz = wi + p[10] * z;                 					//  a_mem
		float xiz = xi + p[2] * z;                  					//  base_mem
		float yiz = yi + p[6] * z;                  					//  thresh
													// (nx,ny)= integer pixel coords of point in new frame
		float nx = xiz / wiz;						// NB Need coords of 4 pixels around the projected point.
		float ny = yiz / wiz;
		if (ny<1 || nx<1 || ny >480 || nx>640) {ny=1; nx=1;} // test for out_of_image

		int nx_ = ceil(nx);
		int ny_ = ceil(ny);

		coff_11 = (ny_ * cols) + nx_;				// offset of pixel in new frame. TODO interpolated sample of pixel.
		coff_10 = (ny_-1) * cols + nx_;
		coff_01 = ny_     * cols + (nx_ -1);
		coff_00 = (ny_-1) * cols + (nx_ -1);
		if (coff_11 > 307200*3) continue; 			// sampled pixel outside of image  TODO check the effect on DTAM's logic of pixels outside the image.

		c_11.x= img[coff_11*3]; c_11.y=img[1+(coff_11*3)], c_11.z=img[2+(coff_11*3)];		// c.xyz = rgb values of pixel in new frame
		c_10.x= img[coff_10*3]; c_10.y=img[1+(coff_10*3)], c_10.z=img[2+(coff_10*3)];
		c_01.x= img[coff_01*3]; c_01.y=img[1+(coff_01*3)], c_01.z=img[2+(coff_01*3)];
		c_00.x= img[coff_00*3]; c_00.y=img[1+(coff_00*3)], c_00.z=img[2+(coff_00*3)];

		float factor_x = fmod(nx,1);
		float factor_y = fmod(ny,1);
		/*float3*/ c =  factor_y * (c_11*factor_x  + c_01*(1-factor_x)) + (1-factor_y) * (c_10*factor_x  + c_00*(1-factor_x));  // bi-linear interpolation

		float thresh = weight;						// occlusionThreshold set at 0.05 by default in main().

		v1 = fabs(c.x - B.x);						// rgb photometric cost: bi-linear. ? should read the keyframe B as a float ?
		v2 = fabs(c.y - B.y);
		v3 = fabs(c.z - B.z);
		del = v1 + v2 + v3;							// L1 norm between keyframe & new frame pixels.
		del = fmin(del/10000, thresh)*3.0f / thresh;		// if(del>0.05) del=3.0, else del=del*60  //  60=3/0.05 //  0.0<=del<=3.0
													// NB divide del by 256 to move from char to float value range.
		if (c.x + c.y + c.z != 0)		{ 			// If new image pixel is NOT black, (i.e. null, out of frame)
			ns = (c0*w + del) / (w + 1);			// c0 = existing value in this costvol elem.
			cdata[offset_1 + z*layerStep] = ns;		// Costdata, same location c0 was read from.  // CostVol set here ###########
			hdata[offset_1 + z*layerStep] = w + 1;	// Weightdata, same location w  was read from.
		}else{
			ns = c0;
		}
		if (ns < minv) { 							// logging intermediate data
			minv = ns;
			mini = z;
		}
		maxv = fmax(ns, maxv);
	}
	// NB All of these are for logging intermediate data for diagnostics, not later use.
	lo[global_id] 	= minv;//base[offset_3];// base[offset_3];//.x;  // export the images as accessed by the kernel.
	a[offset_1] 	= mini;//v1;//
	d[offset_1] 	= ns;//v2;//
	hi[offset_1] 	= maxv;//c_11.x;//v3;//del;//maxv;//img[coff_11*3];  //  export the images as accessed by the kernel.
}// # currently all c.x = 0, ie all pixels are out of the keyframe, even for the first image


 __kernel void CacheG1(__global char* base, __global float* g1p, int cols, int rows)                        // called as "cache1_kernel" in RunCL.cpp
 {
	 /*
	 int global_id = get_global_id(0);
	 //int local_id  = get_local_id(0);
	 int x = global_id; // local_work_size NB dynamic depend on gpu params TODO

	 if(x> cols*rows){
		 printf("\nx> cols*rows ##");
		 return;
	}                       // If out of range, don't process.
	*/
	 int x = get_global_id(0);
	 int y = x / cols;
	 x = x % cols;
	 int upoff = -(y != 0)*cols;                   // up, down, left, right offsets ?
	 int dnoff = (y < rows-1) * cols;
     int lfoff = -(x != 0);
	 int rtoff = (x < cols-1);

	 barrier(CLK_GLOBAL_MEM_FENCE);

	 unsigned int offset = x + y * cols;
	 //if(offset<0 || offset>cols*rows){ printf("\n(offset<0 || offset>cols*rows)##"); return; }

	 float pu, pd, pl, pr;                         // rho, photometric difference: up, down, left, right, of grayscale ref image.
	 float g0x, g0y, g0, g1;

	 pr = base[offset + rtoff];					   // NB base = grayscale CV_8UC1 image.
	 pl = base[offset + lfoff];
	 pu = base[offset + upoff];
	 pd = base[offset + dnoff];

	 g0x = fabs(pr - pl);                          // abs value of vert & horiz gradients.
	 g0y = fabs(pd - pu);
	  
	 g0 = fmax(g0x, g0y);                          // max of vert & horiz gradients.
	 g1 = sqrt(g0);
	 g1 = exp(-3.5 * g1);

	 //if(global_id%10000==0) printf("\nglobal_id=%i, offset=%i, local_id=%i",global_id, offset, local_id);
	 g1p[offset] = g1;                             // save g1 for this pixel.
 }

 __kernel void CacheG2(__global float* g1p, __global float* gxp, __global float* gyp, int cols,int rows)    // called as "cache2_kernel" in RunCL.cpp
 {
	 int x = get_global_id(0);
	 if(x> cols*rows){return;}                     // If out of range, don't process.
	 int y = x / cols;
	 x = x % cols;
	 unsigned int offset = x + y * cols; 
	 
	 int dnoff = (y < rows-1) * cols;              // down and right one pixel offsets, unless at the edge of image.
	 int rtoff = (x < cols-1);
	 float g1h, g1l, g1r, g1u, g1d,gx,gy;

	 barrier(CLK_GLOBAL_MEM_FENCE);

	 g1h = g1p[offset];
	 g1l = g1h;
	 g1u = g1h;
	 g1d = g1p[offset + dnoff];
	 g1r = g1p[offset + rtoff];
	 gx = fmax(g1l, g1r);                          // pick max of this & adj g1 value, for vert & horiz
	 gy = fmax(g1u, g1d);
	
	 gxp[offset] = gx;                             // Save G for this keyframe ref image. 
	 gyp[offset] = gy;
 }

/*
  void fsetPositive(float *fpnum)								// Uses bitwise "inclusive OR"  // does NOT WORK
 {
	 const int positive_zero_int = 0; 							// by ISO C-11 standard, sign-bit=1, followed by 31 zeroes.
	 //float fptemp = *fpnum;
	 //int   intemp = ((int)fptemp);
	 //intemp = ((int)fptemp) | positive_zero_int;
	 *fpnum = (float)(((int)*fpnum) | positive_zero_int);		// by ISO C-11 standard, sign-bit=1, followed 8 expoment bits, then 23 mantissa bits.

	 //*fpnum | (float)0.0; // does this work ?
 }
*/


 __kernel void CacheG3(__global uchar* base, /*__global float* g1p,*/ __global float* gxp, __global float* gyp, int cols, int rows) //, float beta// new version
 {
	 int x = get_global_id(0);
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

	 float g;
	 g = fabs(pr - pl); // ## TODO why is "if(g>64)" needed ? & why this value ? better solutions ?
	 /*if (g>64) gxp[offset] = 0.0; else*/
	 gxp[offset] = g;	//g0x          // abs value of vert & horiz gradients.
	 g = fabs(pd - pu);
	 /*if (g>64) gyp[offset] = 0.0; else*/
	 gyp[offset] = g;	//g0y
	 /*
	 // Huber norm of image gradient, in eq (5).
	 float alpha = -1; 													// in DTAM<1000lines
	 //float beta  = 0,001 // while theta>0.001, else beta = 0.0001 	// from the paper, vs beta=2 in DTAM<1000lines
	 //float epsilon = 0.0001											// in DTAM<1000lines

	 float huber_x = (g0x > beta)*(g0x*g0x/(2*beta)) + (g0x <= beta)*(fabs(g0x) - beta/2); // If weighting remains 2D, need to store direction: (g0x/fabs(g0x))
	 float huber_y = (g0y > beta)*(g0y*g0y/(2*beta)) + (g0y <= beta)*(fabs(g0y) - beta/2); // Otherwise use the norm to merge the dimensions.

	 gxp[offset] = exp(alpha * huber_x);
	 gyp[offset] = exp(alpha * huber_y); // TODO (i) save these to a global variable, (ii) write host fn, and (iii) view images.

	 g1p[offset] = (g0x/fabs(g0x)) * (g0y/fabs(g0y));	// +/-1, holds which diagonal the gradient aligns with.
	 */
	/*
	 if (g>64 && offset%100==0){
		 //float fpnum = (pd-pu);
		 //fsetPositive(&fpnum);
		 int temp = pd-pu;
		 if(temp<0.0) temp *=-1;
		 //const int pos_zero = 0;
		 //temp = temp | pos_zero;
		 //uint utemp1 = (uint)temp;
		 //uint utemp2 = utemp1 <<1;
		 //uint utemp3 = utemp2 >>1;
		 //temp = (int)utemp3;
		 printf("\nCacheG3: g=fabs(pd-pu)=%f, pd=%f, pu=%f, (pd-pu)=%f,  temp=%i ",//, ((pd-pu)|(float)0.0)=%f,  fsetPositive(float *fpnum)=%f",
		 g, pd, pu, (pd-pu), temp);//,  (float)((int)(pd-pu)|(int)0.0),  fpnum );
	 }
	 */
 }



 __kernel void CacheG4( __global float* g1p, __global float* gxp, __global float* gyp, int cols, int rows, float beta)
 {
	 int x = get_global_id(0);
	 int y = x / cols;
	 x = x % cols;
	 if (x<2 || x > cols-2 || y<2 || y>rows-2) return; // needed for wider kernel
	 int upoff = -(y != 0)*cols;                   // up, down, left, right offsets, by boolean logic.
	 int dnoff = (y < rows-1) * cols;
     int lfoff = -(x != 0);
	 int rtoff = (x < cols-1);

	 barrier(CLK_GLOBAL_MEM_FENCE);
	 unsigned int offset = x + y * cols;
	 /*float grad*/ g1p[offset] = 	3*gxp[offset] + gxp[offset + upoff] + gxp[offset + dnoff]  /
									+ 3*gyp[offset] + gyp[offset + rtoff] + gyp[offset + lfoff] ; // sobel edge, from grad_xy

	 //float alpha = -1; 								// in DTAM<1000lines
	 //g1p[offset] = exp(alpha * ((grad > beta)*(grad*grad/(2*beta)) + (grad <= beta)*(fabs(grad) - beta/2)));	// Exp of Huber norm on sobel edges
 }



 
 __kernel void UpdateQ(
	 __global float* gxpt,                                                                // called as "updateQ_kernel" in RunCL.cpp
	 __global float* gypt,                         // gxpt, gypt : G for this keyframe 
	 __global float* gqxpt,
	 __global float* gqypt,                        // gqxpt, gqypt : Q ?
	 __global float* dpt,                          // depth D ?
	 float epsilon,
	 float sigma_q,
	 int cols,
	 int rows)
 {
	 int x = get_global_id(0);
	 int y = x / cols;
	 x = x % cols;
	 int dnoff = (y < rows-1) * cols;              // down offset
	 int rtoff = (x < cols-1);                     // right offset
	 unsigned int pt = x + y * cols;               // index of this pixel

	 barrier(CLK_GLOBAL_MEM_FENCE);
	
	 float dh = dpt[pt];                           // depth 'here'
	 float gqx, gqy, dr, dd, qx, qy, gx, gy;       // GQx,y, D_right, D_down, Q, G ?

	 gqx = gqxpt[pt];                                              // Update gq_x_pt
	 gx = gxpt[pt] + .005f;
	 dr = dpt[pt + rtoff];
	 qx = gqx / gx;
	 qx = (qx + sigma_q*gx*(dr - dh)) / (1 + sigma_q*epsilon);     // (dr - dh) = gradient of depth map
	 qx = qx / fmax(1.0f, fabs(qx));                               // Pi_q(x) function from step 1 in DTAM paper.
	 gqx = gx *  qx;
	 gqxpt[pt] = gqx;

	 gqy = gqypt[pt];                                              // Update gq_y_pt
	 gy = gypt[pt] + .005f;
	 dd = dpt[pt + dnoff];
	 qy = gqy / gy;
	 qy = (qy + sigma_q*gy*(dd - dh)) / (1 + sigma_q*epsilon);
	 qy = qy / fmax(1.0f, fabs(qy));
	 gqy = gy *  qy;
	 gqypt[pt] = gqy;
 }

 
 __kernel void UpdateD(__global float* gqxpt,                                                               // called as "updateD_kernel" in RunCL.cpp 
	 __global float* gqypt,
	 __global float* dpt,
	 __global float* apt,
	 float theta,
	 float sigma_d,
	 int cols)
 {
	 int x = get_global_id(0);
	 int y = x / cols;
	 x = x % cols;
	 int upoff = -(y != 0)*cols;
	 int lfoff = -(x != 0);

	 unsigned int pt = x + y * cols;

	 float dacc;
	 float gqr, gql;

	 barrier(CLK_GLOBAL_MEM_FENCE);

	 gqr = gqxpt[pt];
	 gql = gqxpt[pt + lfoff];
	 dacc = gqr - gql;                                               // depth_accumulator = gq_right - gq_left
	
	 float gqu, gqd;
	 float d = dpt[pt];                                              // Depthmap 'd'
	 float a = apt[pt];                                              // Auxiliary_variable 'a'
	 gqd = gqypt[pt];                                                // gq_down
	 gqu = gqypt[pt + upoff];                                        // gq_up
	 if (y == 0)gqu = 0;
	 dacc += gqd - gqu;                                              // depth_accumulator += gq_down - gq_up
	
	 d = (d + sigma_d*(dacc + a / theta)) / (1 + sigma_d / theta);   // Update 'd' depthmap, from step 1 in DTAM paper.

	 dpt[pt] = d;
 }

 
 float afunc(float data, float theta, float d, float ds, int a, float lambda)                               // used in __kernel void UpdateA(...) below
 {
	 return 1.0 / (2.0*theta)*ds*ds*(d - a)*(d - a) + data * lambda;                     // Eq(14) from the DTAM paper. ds= depthStep= 1.0f / layers
 }

 
 __kernel void UpdateA(
	 __global float* cdata,                                                               // called as "updateA_kernel" in RunCL.cpp
	 __global float* a,
	 __global float* d,
	 int layers,
	 int cols,
	 int rows,
	 float lambda,
	 float theta)
 {
	 int x = get_global_id(0);
	 int y = x / cols;
	 x = x % cols;
	 unsigned int pt = x + y * cols;

	 barrier(CLK_GLOBAL_MEM_FENCE);

	 float dv = d[pt];

	 float depthStep = 1.0f / layers;
	 float vlast, vnext, v, A, B, C;

	 unsigned int mini = 0;
	 unsigned int layerstep = rows * cols;

	 float minv = v = afunc(cdata[pt], theta, dv, depthStep, 0, lambda);                 // apply 'afunc(..)' above.

	 vnext = afunc(cdata[pt + layerstep], theta, dv, depthStep, 1, lambda);

	 for (unsigned int z = 2; z<layers; z++) {                                           // Point-wise exhaustive search: Eq(13) from DTAM paper.
		 vlast = v;                                                                      // arg min (a in Depth), E^aux(u,d_u,a_u)
		 v = vnext;                                                                      // Where E^aux is Eq(14), 'afuc(...)' above.
		 vnext = afunc(cdata[pt + z * layerstep], theta, dv, depthStep, z, lambda);
		 if (v<minv) {
			 A = vlast;
			 C = vnext;
			 minv = v;
			 mini = z - 1;
		 }
	 }

	 if (vnext<minv) {//last was best
		 a[pt] = layers - 1;
		 return;
	 }

	 if (mini == 0) {//first was best
		 a[pt] = 0;
		 return;
	 }

	 B = minv;//avoid divide by zero, since B is already <= others, make < others

	 float denom = (A - 2 * B + C);            // gradient E^aux                         // Single Newton step for refinement of Auxiliary_variable 'a'
	 float delt = (A - C) / (denom * 2);       // grad^2 E^aux                           // See section 2.2.5 of the DTAM paper, 
                                                                                         // "Increasing Solution Accuracy".
	 if (denom != 0)
		 a[pt] = delt + (float)mini;
	 else
		 a[pt] = mini;
 }


 __kernel void InitializeAD(
	 __global float* cdata,                                                               // called as "InitializeAD_kernel" in RunCL.cpp
	 __global float* a,
	 __global float* d,
	 int layers,
	 int layerstep)
 {
	 unsigned int global_id_ = get_global_id(0);
	 if (global_id_ >= layerstep) return;

	 float v, v_next;
	 v=cdata[global_id_];
	 float layer_=0;

	 for (unsigned int z = 1; z<layers; z++) {                                           // Point-wise exhaustive search: for each pixel ray of cost vol
		 v_next = cdata[global_id_ + z * layerstep];
		 if (v_next<v) {
			 v=v_next;
			 layer_=z;
		 }
	 }
	 a[global_id_]=layer_;
	 d[global_id_]=layer_;
 }
