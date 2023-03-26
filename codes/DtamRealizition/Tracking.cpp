

/* NB need separate OpenCL command queues for tracking, mapping, and data upload.
 * see "OpenCL_ Hide data transfer behind GPU Kernels runtime _ by Ravi Kumar _ Medium.mhtml"
 * NB occupancy of the GPU. Need to view with nvprof & Radeon™ GPU Profiler
 *
 * Initially just get the algorithm to work, then optimise data flows, GPU occupancy etc.
 */

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


/////////////////////////////////////////////////////////////






