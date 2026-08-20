#include <string>
#include <vector>
#include <algorithm>
#include <time.h>
#include <list>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <yaml-cpp/yaml.h>

#include <lvt2calib/lvt2Calib.h>
#include <lvt2calib/slamBase.h>

#define DEBUG 0

using namespace std;

string calib_result_dir_ = "", features_info_dir_ = "", calib_result_name_ = "";
string ns_l1, ns_l2, ref_ns;

bool save_calib_file = false, is_multi_exp = false;
int sample_size = 0, sample_num = 0;
ostringstream os_extrinsic_min3d, os_calibfile_log;
list<int> sample_sequence;

lvt2Calib mycalib(L2L_CALIB);

void ExtCalib(pcl::PointCloud<pcl::PointXYZ>::Ptr laser1_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr laser2_cloud)
{
    cout << "********** 2.0 calibration start **********" << endl;
    cout << "<<<<< calibration via min3D " << endl;
    Eigen::Matrix4d Tr_L1toL2_centroid_min3d = mycalib.ExtCalib3D_ceres(laser1_cloud, laser2_cloud);
    cout << "Tr_L1_to_L2_centroid_min_3d = " << "\n" << Tr_L1toL2_centroid_min3d << endl;

    std::vector<double> calib_result_6dof_min3d = eigenMatrix2SixDOF(Tr_L1toL2_centroid_min3d);
    cout << "x, y, z, roll, pitch, yaw = " << endl;
    for(auto it : calib_result_6dof_min3d) cout << it << endl;

    mycalib.transf_3d = Tr_L1toL2_centroid_min3d;

    Eigen::Matrix3d R_min3d;
    R_min3d = Tr_L1toL2_centroid_min3d.block(0,0,3,3);

    cout << "********** 3.0 calculate error **********" << endl;
    cout << "<<<<< 3D Matching Error" << endl;
    vector<double> align_err = mycalib.calAlignError(mycalib.s1_cloud, mycalib.s2_cloud, Tr_L1toL2_centroid_min3d);
    cout << "[rmse_x, rmse_y, rmse_z, rmse_total] = [";
    for(auto it : align_err) cout << it << " ";
    cout << "]" << endl;
    
    if(save_calib_file)
    {
        cout << "********** 4.0 save result **********" << endl;
        std::ofstream savefile_calib_log;

        savefile_calib_log.open(os_calibfile_log.str().c_str(), ios::out|ios::app);
        cout << "<<<<< opening file " << os_calibfile_log.str() << endl;
        savefile_calib_log << currentDateTime() << "," << ref_ns << "," << sample_sequence.size();
        for(auto p : calib_result_6dof_min3d){ savefile_calib_log << "," << p;}
        for(int p = 0; p < 9; p++){ savefile_calib_log << "," << R_min3d(p);}
        for(auto p : align_err){ savefile_calib_log << "," << p;}
        savefile_calib_log << endl;
        savefile_calib_log.close();
        
        std::ofstream of_calib_file;
        of_calib_file.open(os_extrinsic_min3d.str().c_str(), ios::out);
        cout << "<<<<< opening file " << os_extrinsic_min3d.str() << endl;
        of_calib_file << "RT_" + ref_ns << endl;
        for(int i = 0; i < 4; i++)
        {
            for(int j = 0; j < 4; j++)
                of_calib_file << Tr_L1toL2_centroid_min3d(i,j) << ",";
            of_calib_file << endl;
        }
        of_calib_file.close();
        cout << "<<<<< calibration result saved!!!" << endl;
    }
}

void RandSampleCalib(int sample_size_, int sample_num_ = 1)
{
    list<list<int>> sample_sequence_list;
    int total_num = mycalib.feature_points.size();
    if (total_num == sample_size_)
    {
        sample_num_ = 1;
        cout<< "<<<<< use all " << total_num << " positions to do the extrinsic calibration <<<<<" << endl;
    }
    else
        cout<< "<<<<< use " << sample_num_ << " groups of " << sample_size_ << " positions to do the extrinsic calibration <<<<<" << endl;

    RandSample(0, total_num - 1, sample_size_, sample_num_, sample_sequence_list);
    
    int calib_num = 0;
    int sample_list_size = sample_sequence_list.size();
    for(auto p = sample_sequence_list.begin(); p != sample_sequence_list.end(); p++)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr l1_cloud_to_calib (new pcl::PointCloud<pcl::PointXYZ>),
                                            l2_cloud_to_calib (new pcl::PointCloud<pcl::PointXYZ>);
        sample_sequence.clear();
        sample_sequence = *p;
        for (auto pt : *p)
        {
            *l1_cloud_to_calib += *(mycalib.feature_points[pt].sensor1_points);
            *l2_cloud_to_calib += *(mycalib.feature_points[pt].sensor2_points);
        }
        calib_num++;
        cout << "<<<<< Start calibration " << calib_num << "/" << sample_list_size << endl;
        ExtCalib(l1_cloud_to_calib, l2_cloud_to_calib);
    }
}

void fileHandle()
{
    os_extrinsic_min3d.str("");
    os_extrinsic_min3d << calib_result_dir_ << calib_result_name_ << "_exParam" << ".csv";
    os_calibfile_log.str("");
    os_calibfile_log << calib_result_dir_ << "L2L_CalibLog.csv";
    ifstream check_savefile;
    check_savefile.open(os_calibfile_log.str().c_str(), ios::in); 
    if(!check_savefile)
    {
        check_savefile.close();
        ofstream of_savefile;
        of_savefile.open(os_calibfile_log.str().c_str());
        of_savefile << "time,ref,pos_num,x,y,z,r,p,y,R0,R1,R2,R3,R4,R5,R6,R7,R8,align_err_x,align_err_y,align_err_z,align_err_total" << endl;
        of_savefile.close();
    }
    else
    {
        check_savefile.close();
    }
}

int main(int argc, char **argv)
{
    string config_file = "config/calib_l2l.yaml";
    if(argc >= 2) config_file = argv[1];

    YAML::Node cfg;
    try {
        cfg = YAML::LoadFile(config_file);
    } catch(const std::exception& e) {
        cerr << "Failed to load config file '" << config_file << "': " << e.what() << endl;
        return 1;
    }

    calib_result_dir_   = cfg["calib_result_dir"].as<string>("");
    features_info_dir_  = cfg["features_info_dir"].as<string>("");
    calib_result_name_  = cfg["calib_result_name"].as<string>("l2l");
    ns_l1               = cfg["ns_l1"].as<string>("laser1");
    ns_l2               = cfg["ns_l2"].as<string>("laser2");
    save_calib_file     = cfg["save_calib_file"].as<bool>(false);
    is_multi_exp        = cfg["is_multi_exp"].as<bool>(false);
    ref_ns = ns_l1 + "_to_" + ns_l2;

    ostringstream os_in;
    os_in << features_info_dir_;

    cout << "********** Start Calibration **********" << endl;
    cout << "********** 1.0 LOADING DATA **********" << endl;
    if(mycalib.loadCSV(os_in.str().c_str()))
    {
        fileHandle();
        int total_pos_num = mycalib.feature_points.size();
        if(!is_multi_exp)
            RandSampleCalib(total_pos_num, 1);
        else
        {
            for(int i = 1; i <= total_pos_num; i++)
            {
                cout << "<<<<< RandSampleCalib " << i << "/" << total_pos_num << " <<<<<" << endl;
                RandSampleCalib(i, total_pos_num);
            }
        }
    }
    
    return 0;
}
