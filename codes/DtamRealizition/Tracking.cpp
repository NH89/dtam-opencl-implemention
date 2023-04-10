#include "CostVol.h"
using namespace cv;
using namespace std;

/* NB need separate OpenCL command queues for tracking, mapping, and data upload.
 * see "OpenCL_ Hide data transfer behind GPU Kernels runtime _ by Ravi Kumar _ Medium.mhtml"
 * NB occupancy of the GPU. Need to view with nvprof & Radeon™ GPU Profiler
 *
 * Initially just get the algorithm to work, then optimise data flows, GPU occupancy etc.
 */
/*
// 1) make image pyramid & column

// 2) find gaussian & laplacian edges at each level - make shortened list ? (as in Morphogenesis)
//    NB strong edges for anisotropic relaxation
//    Laplacian noise for patch tracking
// NB need to make these available for both tracking & mapping to read.


// 3) find SO3 & SE3 for 1st & 2nd order partial gradients


// 4) dammped least squares tracking, coarse to fine. NB start from previous 6DoF vel & accel
//    NB OpenCL patch size efficiency wrt image vs buffer & global vs local.

//    NB needs to sum the error over the whole image.
//    NB Send tracking result to mapping -> costvol, i.e. can only add cost vol when tracking of frame has been done.
//    but, when modelling rel-vel, reflectance & illum, these change the fit of the cost vol. !!!!!!


// ############
// 5) (i)data exchange between mapping and tracking ?
//    (ii)NB choice of relative scale - how many frames?
//    (iii)Boostrapping ? (a) initially assume spherical middle distance.
//                        (b) initial values of other maps ?

// 6) Need to rotate costvol forward to new frame & decay the old cost.
//    Need to build cost vol around depth range estimate from higher image columnm layer - ie 32 layers in cost vol , not 256 layers
//    therefore cost vol depth varies per pixel.



// 7) Need to be able to add SIRFS regularizations - parsimony & smoothness of both gradient and absolute value, with edges in both.


// 8) add maps and variables to rendering equation


// 9) add/swap cost functions for photometric match - eg feature matching, colour/chroma/luminance, gradient space....




// #############
// Initially just get rigid-world camera tracking, focus on peripheral image for surroundings, rather than deforming object of interest.

// Later add relative velocity map.
*/

/////////////////////////////////////////////////////////////

// # Need a new executable loop, with different name, 
// plus add it to CMakeLists.txt & new kdev launch config.
//




// # create device buffer for image array[6]. NB MipMap needs 1.5 ximage size
// See CostVol::CostVol(..), RunCL::RunCL(..) & RunCL::allocatemem(..)

// from allocatemem(..) adapt for mipmapbuf
dmem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}

status = clEnqueueWriteBuffer(uload_queue, gxmem, 		CL_FALSE, 0, width*height*sizeof(float), 		gx, 			0, NULL, &writeEvt);	
    if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl; exit_(status);}
	
	cvrc.params[ROWS] 			= rows;
	cvrc.params[COLS] 			= cols;
    
    
    
    
    

// # load next image to buffer NB load at position [log_2 index]
// See CostVol::updateCost(..) & RunCL::calcCostVol(..) 


// # convert colour space
//


// # loop kernel for reduction n-times on MipMap
//



// # Get 1st & 2nd order image gradients of MipMap
// see CostVol::cacheGValues(), RunCL::cacheGValue2 & __kernel void CacheG3


// # Predict expected camera motion from previous SE3 vel & accel (both zero initially)
//


// # Get 1st & 2nd order gradients of SO3 wrt predicted pose.
//


// # Predict 1st least squares step of SO3
//


// # Get 1st & 2nd order gradients of SE3 wrt updated pose. (Translation requires depth map, middle depth initally.)
//


// # Predict dammped least squares step of SE3 for whole image + residual of translation for relative velocity map.
//


// # Pass prediction to lower layers. Does it fit better ?
//


// # Repeat SE3 fitting n-times. ? Damping factor adjustment ?
//



//////////////////////////////////////////////////////////////////

// # Get 1st & 2nd order gradients wrt calibration parameters. 
//


// # Take one dammped least squares step of calibration.
//


//////////////////////////////////////////////////////////////////


// # Build depth cost vol on current image, using image array[6] in MipMap buffer, plus RelVelMap, 
// with current camera params & DepthMap if bootstrapping, otherwise with params for each frame.
// NB 2*(1+7) = 14 layers on MipMap DepthCostVol: for model & pyramid, ID cetntre layer plus 7 samples, i.e. centre +&- 3 layers.
// Select naive depth map
// See CostVol::updateCost(..), RunCL::calcCostVol(..) &  __kernel void BuildCostVolume2


// ## Regularize Maps : AbsDepth, GradDepth, SurfNormal, RelVel, 
// # Spatial cost functions
// see CostVol::updateQD(..), RunCL::updateQD(..) & __kernel void UpdateQD(..)


// # Parsimony cost functions : NB Bin sort pixels to find non-spatial neighbours
// see SIFS for priors & Morphogenesis for BinSort


// # Update A : exhaustive search on cost vol with cost fns -> update maps.
// see CostVol::updateA(..), RunCL::updateA(..) & __kernel void UpdateA2(..)



