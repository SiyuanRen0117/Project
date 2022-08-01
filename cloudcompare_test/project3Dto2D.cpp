#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <opencv2/core/core.hpp>
#include <vector>
#include <string>

// void ReadYamlFile(const string &stryamlfile);
using namespace std;
int main (int argc, char** argv)
{
    vector<string> kitti360 = ;
    vector<string> seq = "2013_05_28_drive_0000_sync";
    int seq=0;
    const string sequence = '2013_05_28_drive_%04d_sync'%seq
    string pose_dir = string(root_dir) + string('data_poses') + string (seq);

    

}

class camera 
{


}

class perspectivecamera
{


}


// void  ReadYamlFile(const string &stryamlfile)
// {
//  if (fs::exists(strPathToSequence) == false) {
//     cerr << "FATAL: Could not find the yaml file " << stryamlfile
//          << endl;
//     exit(0);
//   }

// }



