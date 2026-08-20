/*
  lvt2calib - Automatic calibration algorithm for extrinsic parameters of a stereo camera and a velodyne
  Copyright (C) 2017-2018 Jorge Beltran, Carlos Guindel
  This file is part of lvt2calib.
  lvt2calib is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.
  lvt2calib is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with lvt2calib.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
  laser_pattern: Find the circle centers in the laser cloud
*/

#define PCL_NO_PRECOMPILE

#include <ros/ros.h>
#include "ros/package.h"
#include <sensor_msgs/PointCloud2.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PointIndices.h>
#include <pcl_msgs/ModelCoefficients.h>
#include <pcl/common/geometry.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/impl/passthrough.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <dynamic_reconfigure/server.h>
#include <fstream>
#include <iomanip>

#include <lvt2calib/VeloCircleConfig.h>
#include <lvt2calib/ouster_utils.h>
#include <lvt2calib/ClusterCentroids.h>
#include <lvt2calib/offline_circle_centers.h>

using namespace std;
using namespace sensor_msgs;
using namespace pcl;

typedef Ouster::Point PointType;
typedef pcl::PointCloud<PointType> CloudType;

ros::Publisher cumulative_pub, centers_pub, circle_center_pub, centers_centroid_pub, pattern_pub, range_pub, edges_pub, pattern_plane_edges_pub, coeff_pub, aux_pub, auxpoint_pub, debug_pub, xy_cloud_pub, cloud_in_range_pub;
int nFrames; // Used for resetting center computation
pcl::PointCloud<pcl::PointXYZ>::Ptr cumulative_cloud;   // Accumulated centers

// Dynamic parameters
double edge_depth_thre__;
double circle_radius_, circle_radius_thre_,
       centroid_distance_min_, centroid_distance_max_;
Eigen::Vector3f axis_;
double angle_threshold_;
double cluster_size_, cluster_tole_;
double edge_depth_thre_, edge_knn_radius_;
double circle_seg_dis_thre_;
int clouds_proc_ = 0, clouds_used_ = 0;
int min_centers_found_;
int rings_count;

string ns_str;
string json_output_path;

void callback(const PointCloud2::ConstPtr& laser_cloud, const PointCloud2::ConstPtr& calib_cloud)
{
  ROS_DEBUG("[%s/circle] Processing cloud...", ns_str.c_str());

  CloudType::Ptr velo_cloud_pc(new CloudType);
  pcl::PointCloud<pcl::PointXYZI>::Ptr calib_board_pc(new pcl::PointCloud<pcl::PointXYZI>);

  clouds_proc_++;
  fromROSMsg(*laser_cloud, *velo_cloud_pc);
  fromROSMsg(*calib_cloud, *calib_board_pc);

  sensor_msgs::PointCloud2 range_ros;
  pcl::toROSMsg(*calib_board_pc, range_ros);
  range_ros.header = laser_cloud->header;
  range_pub.publish(range_ros);

  sensor_msgs::PointCloud2 cloud_in_range_ros;
  pcl::toROSMsg(*velo_cloud_pc, cloud_in_range_ros);
  cloud_in_range_ros.header = laser_cloud->header;
  cloud_in_range_pub.publish(cloud_in_range_ros);

  lvt2calib::OusterCircleConfig cfg;
  cfg.cluster_size = cluster_size_;
  cfg.min_centers_found = min_centers_found_;
  cfg.rings_count = rings_count;
  cfg.axis = axis_;
  cfg.angle_threshold = angle_threshold_;
  cfg.edge_depth_thre = edge_depth_thre_;
  cfg.edge_knn_radius = edge_knn_radius_;
  cfg.cluster_tole = cluster_tole_;
  cfg.circle_radius = circle_radius_;
  cfg.circle_radius_thre = circle_radius_thre_;
  cfg.circle_seg_dis_thre = circle_seg_dis_thre_;
  cfg.centroid_distance_min = centroid_distance_min_;
  cfg.centroid_distance_max = centroid_distance_max_;

  lvt2calib::OusterCircleExtractionResult extraction;
  if (!lvt2calib::extractOusterCircleFrame(velo_cloud_pc, calib_board_pc, cfg, extraction))
  {
    ROS_WARN("[%s] Not enough centers in current frame", ns_str.c_str());
    return;
  }

  sensor_msgs::PointCloud2 edges_ros;
  pcl::toROSMsg(*extraction.edges_cloud, edges_ros);
  edges_ros.header = laser_cloud->header;
  edges_pub.publish(edges_ros);

  sensor_msgs::PointCloud2 plane_edges_cloud_ros;
  pcl::toROSMsg(*extraction.plane_edges_cloud, plane_edges_cloud_ros);
  plane_edges_cloud_ros.header = laser_cloud->header;
  pattern_plane_edges_pub.publish(plane_edges_cloud_ros);

  sensor_msgs::PointCloud2 pattern_ros;
  pcl::toROSMsg(*extraction.pattern_circles, pattern_ros);
  pattern_ros.header = laser_cloud->header;
  pattern_pub.publish(pattern_ros);

  sensor_msgs::PointCloud2 rotated_pattern_ros;
  pcl::toROSMsg(*extraction.rotated_pattern, rotated_pattern_ros);
  rotated_pattern_ros.header = laser_cloud->header;
  auxpoint_pub.publish(rotated_pattern_ros);

  sensor_msgs::PointCloud2 xy_cloud_ros;
  pcl::toROSMsg(*extraction.xy_cloud, xy_cloud_ros);
  xy_cloud_ros.header = laser_cloud->header;
  xy_cloud_pub.publish(xy_cloud_ros);

  sensor_msgs::PointCloud2 debug_circle_ros;
  pcl::toROSMsg(*extraction.last_circle_inliers, debug_circle_ros);
  debug_circle_ros.header = laser_cloud->header;
  debug_pub.publish(debug_circle_ros);

  sensor_msgs::PointCloud2 ros_circle_center_cloud;
  pcl::toROSMsg(*extraction.frame_centers, ros_circle_center_cloud);
  ros_circle_center_cloud.header = laser_cloud->header;
  circle_center_pub.publish(ros_circle_center_cloud);

  ++nFrames;
  clouds_used_ = nFrames;

  pcl::PointCloud<pcl::PointXYZ>::Ptr centers_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  const bool clustered = lvt2calib::clusterOusterCenters(extraction.frame_centers, cumulative_cloud, cluster_size_, nFrames, centers_cloud);

  sensor_msgs::PointCloud2 ros_pointcloud;
  pcl::toROSMsg(*cumulative_cloud, ros_pointcloud);
  ros_pointcloud.header = laser_cloud->header;
  cumulative_pub.publish(ros_pointcloud);

  pcl_msgs::ModelCoefficients m_coeff;
  pcl_conversions::moveFromPCL(*extraction.plane_coefficients, m_coeff);
  m_coeff.header = laser_cloud->header;
  coeff_pub.publish(m_coeff);

  ROS_INFO("[%s] %d/%d frames: %ld pts in cloud", ns_str.c_str(), clouds_used_, clouds_proc_, cumulative_cloud->points.size());

  if (!clustered)
  {
    ROS_WARN("[%s] Not enough centers after clustering: %ld", ns_str.c_str(), centers_cloud->points.size());
    return;
  }

  sensor_msgs::PointCloud2 ros2_centers_centroid_cloud;
  pcl::toROSMsg(*centers_cloud, ros2_centers_centroid_cloud);
  ros2_centers_centroid_cloud.header = laser_cloud->header;
  centers_centroid_pub.publish(ros2_centers_centroid_cloud);

  lvt2calib::ClusterCentroids to_send;
  to_send.header = laser_cloud->header;
  to_send.cluster_iterations = clouds_used_;
  to_send.total_iterations = clouds_proc_;
  to_send.cloud = ros_circle_center_cloud;
  centers_pub.publish(to_send);
  ROS_INFO("Pattern centers published");

  if (!json_output_path.empty())
  {
    lvt2calib::CircleJsonOptions json_options;
    json_options.sensor_type = "ouster";
    json_options.include_timestamp = true;
    json_options.timestamp = laser_cloud->header.stamp.toSec();
    if (lvt2calib::writeCircleCentersJson(json_output_path, *centers_cloud, json_options))
    {
      ROS_INFO("[%s] Circle centers written to %s", ns_str.c_str(), json_output_path.c_str());
    }
    else
    {
      ROS_WARN("[%s] Could not open JSON output file: %s", ns_str.c_str(), json_output_path.c_str());
    }
  }
}

void param_callback(lvt2calib::VeloCircleConfig &config, uint32_t level){
  circle_radius_ = config.circle_radius;
  ROS_INFO("New pattern circle radius: %f", circle_radius_);
  circle_radius_thre_ = config.circle_radius_thre;
  ROS_INFO("New pattern circle radius threshold: %f", circle_radius_thre_);
  axis_[0] = config.x;
  axis_[1] = config.y;
  axis_[2] = config.z;
  ROS_INFO("New normal axis for plane segmentation: %f, %f, %f", axis_[0], axis_[1], axis_[2]);
  angle_threshold_ = config.angle_threshold;
  ROS_INFO("New angle angle_threshold: %f", angle_threshold_);
  edge_depth_thre_ = config.edge_depth_thre;
  ROS_INFO("New edge_depth_thre: %f", edge_depth_thre_);
  edge_knn_radius_ = config.edge_knn_radius;
  ROS_INFO("New edge_knn_radius: %f", edge_knn_radius_);
  cluster_tole_ = config.cluster_tole;
  ROS_INFO("New cluster_tole: %f", cluster_tole_);
  circle_seg_dis_thre_ = config.circle_seg_dis_thre;
  ROS_INFO("New circle_seg_dis_thre: %f", circle_seg_dis_thre_);
  centroid_distance_min_ = config.centroid_distance_min;
  ROS_INFO("New minimum distance between centroids: %f", centroid_distance_min_);
  centroid_distance_max_ = config.centroid_distance_max;
  ROS_INFO("New maximum distance between centroids: %f", centroid_distance_max_);
}

int main(int argc, char **argv){
  ros::init(argc, argv, "ouster_pattern_circle");
  ros::NodeHandle nh_("~"); // LOCAL
  // ros::Subscriber sub = nh_.subscribe ("cloud1", 1, callback);

  nh_.param("cluster_size", cluster_size_, 0.02);
  nh_.param("min_centers_found", min_centers_found_, 4);
  nh_.param<std::string>("ns", ns_str, "laser");
  nh_.param<std::string>("json_output_path", json_output_path, "/tmp/ouster_circle_centers.json");
  nh_.param("laser_ring_num", rings_count, 32);
  findLaserType(rings_count);

  cloud_in_range_pub = nh_.advertise<PointCloud2>("cloud_in_range", 1);
  range_pub = nh_.advertise<PointCloud2> ("calib_cloud_in", 1);
  edges_pub = nh_.advertise<PointCloud2> ("edges_cloud", 1);
  pattern_plane_edges_pub = nh_.advertise<PointCloud2> ("plane_edges_cloud", 1);


  pattern_pub = nh_.advertise<PointCloud2> ("pattern_circles", 1);
  auxpoint_pub = nh_.advertise<PointCloud2> ("rotated_pattern", 1);
  cumulative_pub = nh_.advertise<PointCloud2> ("cumulative_cloud", 1);
  centers_pub = nh_.advertise<lvt2calib::ClusterCentroids> ("/"+ns_str+"/centers_cloud", 1);
  circle_center_pub = nh_.advertise<PointCloud2> ("circle_center_cloud", 1);
  centers_centroid_pub= nh_.advertise<PointCloud2> ("centers_centroid_cloud", 1);

  debug_pub = nh_.advertise<PointCloud2> ("debug", 1);
  xy_cloud_pub = nh_.advertise<PointCloud2> ("xy_cloud", 1);
  coeff_pub = nh_.advertise<pcl_msgs::ModelCoefficients> ("plane_model", 1);

  nFrames = 0;
  cumulative_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

  dynamic_reconfigure::Server<lvt2calib::VeloCircleConfig> server;
  dynamic_reconfigure::Server<lvt2calib::VeloCircleConfig>::CallbackType f;
  f = boost::bind(param_callback, _1, _2);
  server.setCallback(f);

  message_filters::Subscriber<sensor_msgs::PointCloud2> laser_sub(nh_, "laser_cloud", 10);
  message_filters::Subscriber<sensor_msgs::PointCloud2> calib_sub(nh_, "calib_cloud", 10);
  
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), laser_sub, calib_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));
  
  ros::Rate loop_rate(10);
  bool pause_process = false;
  bool end_process = false;
  bool do_acc_boards = false;
  while(ros::ok())
  {
    // ros::param::get("/pause_process", pause_process);
    ros::param::get("/end_process", end_process);
    ros::param::get("/do_acc_boards", do_acc_boards);
    if(end_process)
    {
      ROS_WARN("[%s/laser_pattern_circle] END......", ns_str.c_str());
      break;
    }
    if(!do_acc_boards)
    {
      ROS_WARN("[%s/laser_pattern_circle] PAUSED......", ns_str.c_str());
      while (!do_acc_boards && ros::ok())
      {
        // ros::param::get("/pause_process", pause_process);
        ros::param::get("/end_process", end_process);
        ros::param::get("/do_acc_boards", do_acc_boards);
        if(end_process)
        {
          ROS_WARN("[%s/laser_pattern_circle] END......", ns_str.c_str());
          break;
        }
      }
      if(end_process)
        break;
      clouds_proc_ = 0;
      clouds_used_ = 0;
      nFrames = 0;
      cumulative_cloud->clear();
    }
    ros::spinOnce();
  }

  ros::shutdown();
  return 0;
}