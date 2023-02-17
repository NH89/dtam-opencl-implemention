// This file converts the file format used at http://www.doc.ic.ac.uk/~ahanda/HighFrameRateTracking/downloads.html
// into the standard [R|T] world -> camera format used by OpenCV
// It is based on a file they provided there, but makes the world coordinate system right handed, with z up, x right, and y forward.
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>

using namespace cv;
using namespace std;
Vec3f direction;
Vec3f upvector;
void convertAhandaPovRayToStandard(const char *filepath,  Mat& R,  Mat& T) // TODO pass by ref intrinsic cameraMatrix
{
    char     text_file_name[600];                                               // open .txt file
    sprintf(text_file_name,"%s",filepath);
    ifstream cam_pars_file(text_file_name);
    if( !cam_pars_file.is_open() ){  cerr<<"Failed to open param file, check location of sample trajectory!"<<endl;  exit(1); }
    char     readlinedata[300];
    Point3d  direction;
    Point3d  upvector;
    Point3d  posvector;
    while(1){
        cam_pars_file.getline(readlinedata,300);                                // read line
        if ( cam_pars_file.eof() ) break;
        istringstream iss;

        if ( strstr(readlinedata,"cam_dir")!= NULL){                            // "cam_dir" direction of optical axis
            string cam_dir_str(readlinedata);
            cam_dir_str = cam_dir_str.substr(cam_dir_str.find("= [")+3);
            cam_dir_str = cam_dir_str.substr(0,cam_dir_str.find("]"));
            iss.str(cam_dir_str);
            iss >> direction.x;    iss.ignore(1,',');
            iss >> direction.z;    iss.ignore(1,',');
            iss >> direction.y;    iss.ignore(1,',');
        }
        if ( strstr(readlinedata,"cam_up")!= NULL){                             // "cam_up" orientation of image wrt world
            string cam_up_str(readlinedata);
            cam_up_str = cam_up_str.substr(cam_up_str.find("= [")+3);
            cam_up_str = cam_up_str.substr(0,cam_up_str.find("]"));
            iss.str(cam_up_str);
            iss >> upvector.x;     iss.ignore(1,',');
            iss >> upvector.z;     iss.ignore(1,',');
            iss >> upvector.y;     iss.ignore(1,',');
        }
        if ( strstr(readlinedata,"cam_pos")!= NULL){                            // "cam_pos" camera position
            string cam_pos_str(readlinedata);
            cam_pos_str = cam_pos_str.substr(cam_pos_str.find("= [")+3);
            cam_pos_str = cam_pos_str.substr(0,cam_pos_str.find("]"));
            iss.str(cam_pos_str);
            iss >> posvector.x;    iss.ignore(1,',');
            iss >> posvector.z;    iss.ignore(1,',');
            iss >> posvector.y;    iss.ignore(1,',');
        }
        // TODO extract camera intrinsic information


    }
    R        = Mat(3,3,CV_64F);                                                 // compute rotation & translation
    R.row(0) = Mat(direction.cross(upvector)).t();
    R.row(1) = Mat(-upvector).t();
    R.row(2) = Mat(direction).t();
    T        = -R*Mat(posvector);

    // TODO comput intrinsic cameraMatrix

   /* cameraMatrix=(Mat_<double>(3,3) << 480,       0.0,    320.    5,
                                           0.0,   480.0,    240.    5,
                                           0.0,     0.0,    1.0     ?  );*/
}

/*
 * Example: "scene_00_0000.txt",   from "ahanda-icl/Trajectory_for_Variable_Frame-Rate/200fps/200fps_GT_archieve/"
cam_pos      = [149.376, 451.41, -285.9]';
cam_dir      = [0.421738, -0.409407, 0.809026]';
cam_up       = [-0.0482194, 0.880868, 0.470899]';
cam_lookat   = [0, 0, 1]';
cam_sky      = [0, 1, 0]';
cam_right    = [1.20423, 0.316017, -0.467833]';
cam_fpoint   = [0, 0, 10]';
cam_angle    = 90;
*/
