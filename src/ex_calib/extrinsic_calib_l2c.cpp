#include <string>
#include <vector>
#include <algorithm>
#include <time.h>
#include <list>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <yaml-cpp/yaml.h>

#include <lvt2calib/lvt2Calib.h>
#include <lvt2calib/slamBase.h>

#define DEBUG 0

using namespace std;

string camera_info_dir_ = "", calib_result_dir_ = "", features_info_dir_ = "", calib_result_name_ = "";
string ns_l, ns_c, ref_ns;
cv::Mat cameraMatrix = Mat_<double>(3,3);

bool save_calib_file = false, is_multi_exp = false;
bool points_visual_on = false;

std::vector<cv::Point2f> lv_2d_projected_min2d, lv_2d_projected_min3d;
int sample_size = 0, sample_num = 0;
ostringstream os_calibfile_log, os_extrinsic_min3d, os_extrinsic_min2d;

list<int> sample_sequence;

lvt2Calib mycalib(L2C_CALIB);

void ExtCalib(pcl::PointCloud<pcl::PointXYZ>::Ptr laser_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr camera_cloud, std::vector<cv::Point2f> cam_2d_sorted)
{
    cout << "********** 2.0 calibration start **********" << endl;
    cout << "<<<<< 2.1 calibration via min3D " << endl;
    Eigen::Matrix4d Tr_l2s_centroid_min3d = mycalib.ExtCalib3D_ceres(laser_cloud, camera_cloud);
    Eigen::Matrix4d Tr_s2c_centroid, Tr_l2c_centroid_min3d;
    Tr_s2c_centroid <<   0, -1, 0, 0,
    0, 0, -1, 0,
    1, 0, 0, 0,
    0, 0, 0, 1;
    Tr_l2c_centroid_min3d = Tr_s2c_centroid * Tr_l2s_centroid_min3d;
    cout << "Tr_laser_to_cam_centroid_min_3d = " << "\n" << Tr_l2c_centroid_min3d << endl;

    std::vector<double> calib_result_6dof_min3d = eigenMatrix2SixDOF(Tr_l2c_centroid_min3d);
    cout << "x, y, z, roll, pitch, yaw = " << endl;
    for(auto it : calib_result_6dof_min3d) cout << it << endl;

    mycalib.transf_3d = Tr_l2s_centroid_min3d;

    cout << "<<<<< 2.2 calibration via min2D " << endl;
    Eigen::Matrix4d Tr_l2c_centroid_min2d = mycalib.ExtCalib2D(laser_cloud, cam_2d_sorted, Tr_l2c_centroid_min3d);
    cout << "Tr_laser_to_cam_min_2d = " << "\n" << Tr_l2c_centroid_min2d << endl;
    Eigen::Matrix4d Tr_l2s_centroid_min2d = Tr_s2c_centroid.inverse() * Tr_l2c_centroid_min2d;
    std::vector<double> calib_result_6dof_min2d = eigenMatrix2SixDOF(Tr_l2c_centroid_min2d);
    cout << "x, y, z, roll, pitch, yaw = " << endl;
    for(auto it : calib_result_6dof_min2d) cout << it << endl;

    mycalib.transf_2d = Tr_l2c_centroid_min2d;

    cout << "********** 3.0 calculate error **********" << endl;
    Eigen::Matrix3d R_min3d, R_min2d;
    R_min3d = Tr_l2c_centroid_min3d.block(0,0,3,3);
    R_min2d = Tr_l2c_centroid_min2d.block(0,0,3,3);

    cout << "<<<<< 3.1 3D Matching Error" << endl;
    vector<double> align_err_min3d = mycalib.calAlignError(mycalib.s1_cloud, mycalib.s2_cloud, Tr_l2s_centroid_min3d);
    vector<double> align_err_min2d = mycalib.calAlignError(mycalib.s1_cloud, mycalib.s2_cloud, Tr_l2s_centroid_min2d);
    cout << "min3d [rmse_x, rmse_y, rmse_z, rmse_total] = [";
    for(auto it : align_err_min3d) cout << it << " ";
    cout << "]" << endl;
    cout << "min2d [rmse_x, rmse_y, rmse_z, rmse_total] = [";
    for(auto it : align_err_min2d) cout << it << " ";
    cout << "]" << endl;

    cout << "<<<<< 3.2 2D Re-projection Error" << endl;
    cv::Mat Tr_l2c_min3d_cv;
    eigen2cv(Tr_l2c_centroid_min3d, Tr_l2c_min3d_cv);
    lv_2d_projected_min3d.clear();
    projectVelo2Cam(mycalib.s1_cloud, cameraMatrix, Tr_l2c_min3d_cv, lv_2d_projected_min3d);
    std::vector<double> rmse_2d_reproj_wt_centroid_min3d = calculateRMSE(mycalib.cam_2d_points, lv_2d_projected_min3d);
    cout << "min3d [rmse_2d_reproj_u, rmse_2d_reproj_v, rmse_2d_reproj_total] = \n[";
    for(auto it : rmse_2d_reproj_wt_centroid_min3d) cout << it << " ";
    cout << "]" << endl;

    cv::Mat Tr_l2c_min2d_cv;
    eigen2cv(Tr_l2c_centroid_min2d, Tr_l2c_min2d_cv);
    lv_2d_projected_min2d.clear();
    projectVelo2Cam(mycalib.s1_cloud, cameraMatrix, Tr_l2c_min2d_cv, lv_2d_projected_min2d);
    std::vector<double> rmse_2d_reproj_wt_centroid_min2d = calculateRMSE(mycalib.cam_2d_points, lv_2d_projected_min2d);
    cout << "min2d [rmse_2d_reproj_u, rmse_2d_reproj_v, rmse_2d_reproj_total] = \n[";
    for(auto it : rmse_2d_reproj_wt_centroid_min2d) cout << it << " ";
    cout << "]" << endl;
    
    if(save_calib_file)
    {
        cout << "********** 4.0 save calibration result **********" << endl;
        std::ofstream savefile_calib_log;

        savefile_calib_log.open(os_calibfile_log.str().c_str(), ios::out|ios::app);
        cout << "<<<<< opening file " << os_calibfile_log.str() << endl;
        savefile_calib_log << currentDateTime() << "," << ref_ns+"_min3d" << "," << sample_sequence.size();
        for(auto p : calib_result_6dof_min3d){  savefile_calib_log << "," << p;}
        for(int i = 0; i < 9; i++){ savefile_calib_log << "," << R_min3d(i);}
        for(auto p : align_err_min3d){ savefile_calib_log << "," << p;}
        for(auto p : rmse_2d_reproj_wt_centroid_min3d){ savefile_calib_log << "," << p;}
        savefile_calib_log << endl;
        savefile_calib_log.close();

        savefile_calib_log.open(os_calibfile_log.str().c_str(), ios::out|ios::app);
        savefile_calib_log << currentDateTime() << "," << ref_ns+"_min2d" << "," << sample_sequence.size();
        for(auto p : calib_result_6dof_min2d){  savefile_calib_log << "," << p;}
        for(int p = 0; p < 9; p++){ savefile_calib_log << "," << R_min2d(p);}
        for(auto p : align_err_min2d){ savefile_calib_log << "," << p;}
        for(auto p : rmse_2d_reproj_wt_centroid_min2d){ savefile_calib_log << "," << p;}
        savefile_calib_log << endl;
        savefile_calib_log.close();
        
        std::ofstream savefile_exparam;
        savefile_exparam.open(os_extrinsic_min3d.str().c_str(), ios::out);
        cout << "<<<<< opening file " << os_extrinsic_min3d.str() << endl;
        savefile_exparam << "RT_" + ref_ns + "_min3d" << endl;
        for(int i = 0; i < 4; i++)
        {
            for(int j = 0; j < 4; j++)
                savefile_exparam << Tr_l2c_centroid_min3d(i,j) << ", ";
            savefile_exparam << endl;
        }
        savefile_exparam.close();

        savefile_exparam.open(os_extrinsic_min2d.str().c_str(), ios::out);
        cout << "<<<<< opening file " << os_extrinsic_min2d.str() << endl;
        savefile_exparam << "RT_" + ref_ns + "_min2d" << endl;
        for(int i = 0; i < 4; i++)
        {
            for(int j = 0; j < 4; j++)
                savefile_exparam << Tr_l2c_centroid_min2d(i,j) << ", ";
            savefile_exparam << endl;
        }
        savefile_exparam.close();
        cout << "<<<<< calibration result saved!!!" << endl;
    }
    
    if(points_visual_on)
    {
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_min3d(new pcl::visualization::PCLVisualizer("3d matching viewer (min3d)"));
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_min2d(new pcl::visualization::PCLVisualizer("3d matching viewer (min2d)"));
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_raw(new pcl::visualization::PCLVisualizer("raw viewer"));

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr laser_cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>), camera_cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>), laser_cloud_camera_min3d_rgb(new pcl::PointCloud<pcl::PointXYZRGB>), laser_cloud_camera_min2d_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);

        pcl::copyPointCloud(*mycalib.s1_cloud, *laser_cloud_rgb);
        pcl::copyPointCloud(*mycalib.s2_cloud, *camera_cloud_rgb);
        int point_size = laser_cloud_rgb->points.size();
        int color_gap = 255 / point_size;
        for (int i = 0; i < point_size; ++i)
        {
            int color_value = (i + 1) * color_gap;
            (laser_cloud_rgb->points.begin() + i)->r = color_value;
            (camera_cloud_rgb->points.begin() + i)->g = color_value;
        }

        Eigen::Matrix4f Tr_l2s_min3d_f = Tr_l2s_centroid_min3d.cast<float>();
        Eigen::Matrix4f Tr_l2s_min2d_f = Tr_l2s_centroid_min2d.cast<float>();
        pcl::transformPointCloud(*laser_cloud_rgb, *laser_cloud_camera_min3d_rgb, Tr_l2s_min3d_f);
        pcl::transformPointCloud(*laser_cloud_rgb, *laser_cloud_camera_min2d_rgb, Tr_l2s_min2d_f);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr matching_pc_min3d(new pcl::PointCloud<pcl::PointXYZRGB>), matching_pc_min2d(new pcl::PointCloud<pcl::PointXYZRGB>), raw_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
        *matching_pc_min3d = *camera_cloud_rgb + *laser_cloud_camera_min3d_rgb;
        *matching_pc_min2d = *camera_cloud_rgb + *laser_cloud_camera_min2d_rgb;
        *raw_pc = *camera_cloud_rgb + *laser_cloud_rgb;

        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_handler_min3d(matching_pc_min3d);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_handler_min2d(matching_pc_min2d);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_handler_raw(raw_pc);

        viewer_min3d->setBackgroundColor(255, 255, 255);
        viewer_min2d->setBackgroundColor(255, 255, 255);
        viewer_raw->setBackgroundColor(255, 255, 255);
        viewer_min3d->addPointCloud<pcl::PointXYZRGB>(matching_pc_min3d, rgb_handler_min3d, "matching_pc_min3d");
        viewer_min2d->addPointCloud<pcl::PointXYZRGB>(matching_pc_min2d, rgb_handler_min2d, "matching_pc_min2d");
        viewer_raw->addPointCloud<pcl::PointXYZRGB>(raw_pc, rgb_handler_raw, "raw_pc");
        viewer_min3d->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "matching_pc_min3d");
        viewer_min2d->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "matching_pc_min2d");
        viewer_raw->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "raw_pc");
        viewer_min3d->addCoordinateSystem(1.0);
        viewer_min2d->addCoordinateSystem(1.0);
        viewer_raw->addCoordinateSystem(1.0);

        while(!viewer_min3d->wasStopped() && !viewer_min2d->wasStopped() && !viewer_raw->wasStopped())
        {
            viewer_min2d->spinOnce(100);
            viewer_min3d->spinOnce(100);
            viewer_raw->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
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
        pcl::PointCloud<pcl::PointXYZ>::Ptr l_cloud_to_calib (new pcl::PointCloud<pcl::PointXYZ>),
                                            c_cloud_to_calib (new pcl::PointCloud<pcl::PointXYZ>);
        vector<cv::Point2f> cam_2d_to_calib;
        sample_sequence.clear();
        sample_sequence = *p;
        for (auto pt : *p)
        {
            *l_cloud_to_calib += *(mycalib.feature_points[pt].sensor1_points);
            *c_cloud_to_calib += *(mycalib.feature_points[pt].sensor2_points);
            cam_2d_to_calib.insert(cam_2d_to_calib.end(), mycalib.feature_points[pt].camera_2d.begin(), mycalib.feature_points[pt].camera_2d.end());
        }
        calib_num++;
        cout << "<<<<< Start calibration " << calib_num << "/" << sample_list_size << endl;
        ExtCalib(l_cloud_to_calib, c_cloud_to_calib, cam_2d_to_calib);
    }
}

void fileHandle()
{
    os_extrinsic_min3d.str("");
    os_extrinsic_min2d.str("");
    os_extrinsic_min3d << calib_result_dir_ << calib_result_name_ << "_exParam_min3d" << ".csv";
    os_extrinsic_min2d << calib_result_dir_ << calib_result_name_ << "_exParam_min2d" << ".csv";

    os_calibfile_log.str("");
    os_calibfile_log << calib_result_dir_ << "L2C_CalibLog.csv";
    ifstream check_savefile;
    check_savefile.open(os_calibfile_log.str().c_str(), ios::in); 
    if(!check_savefile)
    {
        check_savefile.close();
        ofstream of_savefile;
        of_savefile.open(os_calibfile_log.str().c_str());
        of_savefile << "time,ref,pos_num,x,y,z,r,p,y,R0,R1,R2,R3,R4,R5,R6,R7,R8,align_err_x,align_err_y,align_err_z,align_err_total,rmse_2d_reproj_u,rmse_2d_reproj_v,rmse_2d_reproj_total" << endl;
        of_savefile.close();
    }
    else
    {
        check_savefile.close();
    }
}


int main(int argc, char **argv)
{
    string config_file = "config/calib_l2c.yaml";
    if(argc >= 2) config_file = argv[1];

    YAML::Node cfg;
    try {
        cfg = YAML::LoadFile(config_file);
    } catch(const std::exception& e) {
        cerr << "Failed to load config file '" << config_file << "': " << e.what() << endl;
        return 1;
    }

    calib_result_dir_   = cfg["calib_result_dir"].as<string>("");
    camera_info_dir_    = cfg["camera_info_dir"].as<string>("");
    features_info_dir_  = cfg["features_info_dir"].as<string>("");
    calib_result_name_  = cfg["calib_result_name"].as<string>("l2c");
    ns_l                = cfg["ns_l"].as<string>("laser");
    ns_c                = cfg["ns_c"].as<string>("cam");
    save_calib_file     = cfg["save_calib_file"].as<bool>(false);
    is_multi_exp        = cfg["is_multi_exp"].as<bool>(false);
    points_visual_on    = cfg["points_visual_on"].as<bool>(false);

    ref_ns = ns_l + "_to_" + ns_c;
    ostringstream os_in;
    os_in << features_info_dir_;

    cout << "********** Start Calibration **********" << endl;
    cout << "********** 1.0 LOADING DATA **********" << endl;
    if(mycalib.loadCSV(os_in.str().c_str()))
    {
        std::ostringstream oss_CamIntrinsic;
        oss_CamIntrinsic << camera_info_dir_;
        ParameterReader pr_cam_intrinsic(oss_CamIntrinsic.str());
        cameraMatrix = pr_cam_intrinsic.ReadMatFromTxt(pr_cam_intrinsic.getData("K"),3,3);
        cv::cv2eigen(cameraMatrix, mycalib.cameraMatrix_);
        cout << "cameraMatrix: \n" << cameraMatrix << endl;

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
