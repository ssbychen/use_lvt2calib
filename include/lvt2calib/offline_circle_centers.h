#ifndef LVT2CALIB_OFFLINE_CIRCLE_CENTERS_H
#define LVT2CALIB_OFFLINE_CIRCLE_CENTERS_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include <ros/ros.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <lvt2calib/AutoDetectLaser.h>
#include <lvt2calib/FourCircleCenters.h>
#include <lvt2calib/ouster_utils.h>

namespace lvt2calib {

struct LaserPatternConfig {
  bool use_RG_Pseg = false;
  bool use_vox_filter = true;
  bool use_i_filter = true;
  bool use_gauss_filter = true;
  bool use_gauss_filter2 = true;
  bool use_statistic_filter = false;
  bool auto_mode = true;
  bool is_gazebo = false;

  double remove_x_min = -1.0;
  double remove_x_max = 1.0;
  double voxel_grid_size = 0.01;
  double gauss_k_sigma = 3.0;
  double gauss_k_thre_rt_sigma = 3.0;
  double gauss_k_thre = 0.05;
  double gauss_conv_radius = 0.02;
  double cluster_tole = 0.08;
  double cluster_size_min = 0.03;
  double cluster_size_max = 1.5;
  double Pseg_dis_thre = 0.02;
  int Pseg_iter_num = 1000;
  double Pseg_size_min = 0.1;
  double RG_smooth_thre_deg = 5.0;
  double RG_curve_thre = 0.1;
  int RG_neighbor_n = 30;
  int sor_MeanK = 10;
  int sor_StddevMulThresh = 1;
  double i_filter_out_min = 0.0;
  double i_filter_out_max = 30.0;
  double boundEstRad = 30.0;
  double normEstRad = 50.0;
  double rmse_ukn2tpl_thre = 0.04;
  double rmse_tpl2ukn_thre = 0.03;
  double circle_radius = 0.12;
  double circle_seg_thre = 0.02;
  double centroid_dis_min = 0.15;
  double centroid_dis_max = 0.25;
  int min_centers_found = 4;
};

struct OusterCircleConfig {
  double cluster_size = 0.1;
  int min_centers_found = 4;
  int rings_count = 32;
  Eigen::Vector3f axis = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
  double angle_threshold = 0.55;
  double edge_depth_thre = 0.5;
  double edge_knn_radius = 0.1;
  double cluster_tole = 0.55;
  double circle_radius = 0.12;
  double circle_radius_thre = 0.02;
  double circle_seg_dis_thre = 0.04;
  double centroid_distance_min = 0.12;
  double centroid_distance_max = 0.45;
};

struct CircleJsonOptions {
  std::string sensor_type;
  bool include_timestamp = false;
  double timestamp = 0.0;
  bool include_input_pcd = false;
  std::string input_pcd;
};

struct OusterCircleExtractionResult {
  pcl::PointCloud<pcl::PointXYZ>::Ptr frame_centers;
  pcl::PointCloud<Ouster::Point>::Ptr edges_cloud;
  pcl::PointCloud<Ouster::Point>::Ptr plane_edges_cloud;
  pcl::PointCloud<pcl::PointXYZ>::Ptr pattern_circles;
  pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_pattern;
  pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud;
  pcl::PointCloud<pcl::PointXYZ>::Ptr last_circle_inliers;
  pcl::ModelCoefficients::Ptr plane_coefficients;

  OusterCircleExtractionResult()
      : frame_centers(new pcl::PointCloud<pcl::PointXYZ>),
        edges_cloud(new pcl::PointCloud<Ouster::Point>),
        plane_edges_cloud(new pcl::PointCloud<Ouster::Point>),
        pattern_circles(new pcl::PointCloud<pcl::PointXYZ>),
        rotated_pattern(new pcl::PointCloud<pcl::PointXYZ>),
        xy_cloud(new pcl::PointCloud<pcl::PointXYZ>),
        last_circle_inliers(new pcl::PointCloud<pcl::PointXYZ>),
        plane_coefficients(new pcl::ModelCoefficients) {}
};

static constexpr double kLivoxMaxSize = 120.0 * 80.0;
static constexpr double kRepetitiveMaxSize = 120.0 * 2.0 / 2.0 + (120.0 - 24.0 * 2.0) * 4.0 / 2.0;

inline LaserPatternConfig makeLivoxPreset(const std::string& preset) {
  LaserPatternConfig cfg;
  (void)preset;
  return cfg;
}

inline LaserPatternConfig makeOusterPreset(const std::string& preset) {
  LaserPatternConfig cfg;
  cfg.use_vox_filter = false;
  cfg.use_gauss_filter = false;
  cfg.use_gauss_filter2 = false;
  cfg.use_statistic_filter = false;
  cfg.cluster_tole = 0.05;
  cfg.cluster_size_min = 0.01;
  cfg.cluster_size_max = 5.0;
  cfg.rmse_ukn2tpl_thre = 0.04;
  cfg.rmse_tpl2ukn_thre = 0.04;
  cfg.i_filter_out_max = 30.0;

  if (preset == "os1_32") {
    cfg.cluster_tole = 0.15;
    cfg.cluster_size_min = 0.8;
    cfg.cluster_size_max = 3.0;
    cfg.i_filter_out_max = 9.0;
  } else if (preset == "os1_64") {
    cfg.cluster_tole = 0.10;
    cfg.cluster_size_min = 0.8;
    cfg.cluster_size_max = 3.0;
    cfg.i_filter_out_max = 10.0;
    cfg.rmse_ukn2tpl_thre = 0.03;
    cfg.rmse_tpl2ukn_thre = 0.03;
  } else if (preset == "os1_128") {
    cfg.cluster_tole = 0.05;
    cfg.cluster_size_min = 0.8;
    cfg.cluster_size_max = 3.0;
    cfg.i_filter_out_max = 80.0;
    cfg.rmse_ukn2tpl_thre = 0.03;
    cfg.rmse_tpl2ukn_thre = 0.03;
  }

  return cfg;
}

inline OusterCircleConfig makeOusterCirclePreset(const std::string& preset) {
  OusterCircleConfig cfg;
  if (preset == "os1_64") {
    cfg.rings_count = 64;
  } else if (preset == "os1_128") {
    cfg.rings_count = 128;
  } else {
    cfg.rings_count = 32;
  }
  return cfg;
}

inline void configureAutoDetectLaser(AutoDetectLaser& detector, const LaserPatternConfig& cfg, double max_size) {
  detector.setRemoveRangeX(cfg.remove_x_min, cfg.remove_x_max);
  detector.useVoxelFilter(cfg.use_vox_filter);
  detector.setVoxelFilterSize(cfg.voxel_grid_size);
  detector.setAutoMode(cfg.auto_mode);
  detector.useGaussFilter(cfg.use_gauss_filter);
  detector.setGaussFilterParam(cfg.gauss_k_sigma, cfg.gauss_k_thre_rt_sigma, cfg.gauss_k_thre, cfg.gauss_conv_radius);
  detector.useStatisticalFilter(cfg.use_statistic_filter);
  detector.setStatisticalFilterParam(cfg.sor_MeanK, cfg.sor_StddevMulThresh);
  detector.setClusterParam(cfg.cluster_tole, cfg.cluster_size_min * max_size, cfg.cluster_size_max * max_size);
  detector.setPlaneSegmentationParam(cfg.Pseg_dis_thre, cfg.Pseg_size_min, cfg.Pseg_iter_num);
  detector.setRGPlaneSegmentationParam(cfg.RG_smooth_thre_deg, cfg.RG_curve_thre, cfg.RG_neighbor_n);
  detector.useIntensityFilter(cfg.use_i_filter);
  detector.setIntensityFilterParam(cfg.i_filter_out_min, cfg.i_filter_out_max);
  detector.setBoundEstKSearch(cfg.boundEstRad);
  detector.setNormEstKSearch(cfg.normEstRad);
  detector.setDiffRMSEThreshold(cfg.rmse_ukn2tpl_thre, cfg.rmse_tpl2ukn_thre);
}

inline void configureFourCircleCenters(FourCircleCenters& detector, const LaserPatternConfig& cfg) {
  detector.setCircleSegDistanceThreshold(cfg.circle_seg_thre);
  detector.setCircleRadius(cfg.circle_radius);
  detector.setCentroidDis(cfg.centroid_dis_min, cfg.centroid_dis_max);
  detector.setMinNumCentersFound(cfg.min_centers_found);
}

inline bool orderPatternCentersYZ(const pcl::PointCloud<pcl::PointXYZ>& cloud, std::vector<pcl::PointXYZ>& ordered) {
  ordered.clear();
  if (cloud.points.size() != 4) {
    return false;
  }

  ordered.resize(4);
  double avg_y = 0.0, avg_z = 0.0;
  for (const auto& pt : cloud.points) {
    avg_y += pt.y;
    avg_z += pt.z;
  }
  const double center_y = avg_y / 4.0;
  const double center_z = avg_z / 4.0;

  for (const auto& pt : cloud.points) {
    const double y_dif = pt.y - center_y;
    const double z_dif = pt.z - center_z;
    if (y_dif > 0 && z_dif > 0) {
      ordered[0] = pt;
    } else if (y_dif < 0 && z_dif > 0) {
      ordered[1] = pt;
    } else if (y_dif < 0 && z_dif < 0) {
      ordered[2] = pt;
    } else {
      ordered[3] = pt;
    }
  }

  return true;
}

inline pcl::PointCloud<pcl::PointXYZ>::Ptr toXYZCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(cloud, *out);
  return out;
}

inline pcl::PointCloud<pcl::PointXYZ>::Ptr orderedCentersCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::PointXYZ> ordered;
  if (orderPatternCentersYZ(cloud, ordered)) {
    out->points.assign(ordered.begin(), ordered.end());
  } else {
    *out = cloud;
  }
  out->width = out->points.size();
  out->height = 1;
  out->is_dense = cloud.is_dense;
  return out;
}

inline bool writeCircleCentersJson(const std::string& output_path,
                                   const pcl::PointCloud<pcl::PointXYZ>& centers,
                                   const CircleJsonOptions& options) {
  std::ofstream jf(output_path.c_str());
  if (!jf.is_open()) {
    return false;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr ordered = orderedCentersCloud(centers);
  jf << std::fixed << std::setprecision(6);
  jf << "{\n";
  jf << "  \"sensor_type\": \"" << options.sensor_type << "\",\n";
  jf << "  \"timestamp\": ";
  if (options.include_timestamp) {
    jf << options.timestamp;
  } else {
    jf << "null";
  }
  jf << ",\n";
  if (options.include_input_pcd) {
    jf << "  \"input_pcd\": \"" << options.input_pcd << "\",\n";
  }
  jf << "  \"num_centers\": " << ordered->points.size() << ",\n";
  jf << "  \"centers\": [\n";
  for (size_t ci = 0; ci < ordered->points.size(); ++ci) {
    const auto& pt = ordered->points[ci];
    jf << "    {\"x\": " << pt.x << ", \"y\": " << pt.y << ", \"z\": " << pt.z << "}";
    if (ci + 1 < ordered->points.size()) {
      jf << ",";
    }
    jf << "\n";
  }
  jf << "  ]\n";
  jf << "}\n";
  return true;
}

inline bool writeCircleCentersJson(const std::string& output_path,
                                   const pcl::PointCloud<pcl::PointXYZI>& centers,
                                   const CircleJsonOptions& options) {
  return writeCircleCentersJson(output_path, *toXYZCloud(centers), options);
}

inline bool extractOusterCircleFrame(const pcl::PointCloud<Ouster::Point>::Ptr& velo_cloud_pc_in,
                                     const pcl::PointCloud<pcl::PointXYZI>::Ptr& calib_board_pc_in,
                                     const OusterCircleConfig& cfg,
                                     OusterCircleExtractionResult& result) {
  findLaserType(cfg.rings_count);

  pcl::PointCloud<Ouster::Point>::Ptr velo_cloud_pc(new pcl::PointCloud<Ouster::Point>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr calib_board_pc(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::copyPointCloud(*velo_cloud_pc_in, *velo_cloud_pc);
  pcl::copyPointCloud(*calib_board_pc_in, *calib_board_pc);

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::SACSegmentation<pcl::PointXYZI> plane_segmentation;
  plane_segmentation.setModelType(pcl::SACMODEL_PARALLEL_PLANE);
  plane_segmentation.setDistanceThreshold(0.01);
  plane_segmentation.setMethodType(pcl::SAC_RANSAC);
  plane_segmentation.setAxis(cfg.axis);
  plane_segmentation.setEpsAngle(cfg.angle_threshold);
  plane_segmentation.setOptimizeCoefficients(true);
  plane_segmentation.setMaxIterations(1000);
  plane_segmentation.setInputCloud(calib_board_pc);
  plane_segmentation.segment(*inliers, *coefficients);

  if (inliers->indices.empty()) {
    return false;
  }

  Eigen::VectorXf coefficients_v(4);
  coefficients_v(0) = coefficients->values[0];
  coefficients_v(1) = coefficients->values[1];
  coefficients_v(2) = coefficients->values[2];
  coefficients_v(3) = coefficients->values[3];

  std::vector<int> indices_f1, indices_f2;
  pcl::PointCloud<Ouster::Point>::Ptr velo_cloud_pc_valid(new pcl::PointCloud<Ouster::Point>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr calib_board_pc_valid(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::removeNaNFromPointCloud(*velo_cloud_pc, *velo_cloud_pc_valid, indices_f1);
  pcl::removeNaNFromPointCloud(*calib_board_pc, *calib_board_pc_valid, indices_f2);
  pcl::copyPointCloud(*velo_cloud_pc_valid, *velo_cloud_pc);
  pcl::copyPointCloud(*calib_board_pc_valid, *calib_board_pc);

  pcl::PointCloud<pcl::PointXYZ>::Ptr calib_board_pc_copy(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(*calib_board_pc, *calib_board_pc_copy);
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(calib_board_pc_copy);

  result.edges_cloud->clear();
  for (auto pt = velo_cloud_pc->points.begin(); pt < velo_cloud_pc->points.end(); ++pt) {
    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;
    pcl::PointXYZ searchP;
    pcl::copyPoint(*pt, searchP);
    if (kdtree.nearestKSearch(searchP, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
      if (pointNKNSquaredDistance[0] <= cfg.edge_knn_radius && pt->intensity > cfg.edge_depth_thre) {
        result.edges_cloud->push_back(*pt);
      }
    }
  }

  if (result.edges_cloud->points.empty()) {
    return false;
  }

  pcl::SampleConsensusModelPlane<Ouster::Point>::Ptr plane_model(
      new pcl::SampleConsensusModelPlane<Ouster::Point>(result.edges_cloud));
  std::vector<int> inliers2;
  plane_model->selectWithinDistance(coefficients_v, .05, inliers2);
  result.plane_edges_cloud->clear();
  pcl::copyPointCloud<Ouster::Point>(*result.edges_cloud, inliers2, *result.plane_edges_cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr circles_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<std::vector<Ouster::Point*> > rings2 = Ouster::getRings(*result.plane_edges_cloud, laser_type);
  int ringsWithCircle = 0;
  for (auto ring = rings2.begin(); ring < rings2.end(); ++ring) {
    if (ring->size() < 4) {
      ring->clear();
      continue;
    }
    ringsWithCircle++;
    ring->erase(ring->begin());
    ring->pop_back();
    for (auto pt = ring->begin(); pt < ring->end(); ++pt) {
      pcl::PointXYZ point;
      point.x = (*pt)->x;
      point.y = (*pt)->y;
      point.z = (*pt)->z;
      circles_cloud->push_back(point);
    }
  }

  if (circles_cloud->points.empty() || circles_cloud->points.size() > static_cast<size_t>(ringsWithCircle * 4)) {
    return false;
  }
  *result.pattern_circles = *circles_cloud;

  pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  Eigen::Vector3f xy_plane_normal_vector(0.0f, 0.0f, -1.0f);
  Eigen::Vector3f floor_plane_normal_vector(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
  Eigen::Affine3f rotation = getRotationMatrix(floor_plane_normal_vector, xy_plane_normal_vector);
  pcl::transformPointCloud(*circles_cloud, *xy_cloud, rotation);

  pcl::PointCloud<pcl::PointXYZ>::Ptr aux_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointXYZ aux_point;
  aux_point.x = 0;
  aux_point.y = 0;
  aux_point.z = (-coefficients_v(3) / coefficients_v(2));
  aux_cloud->push_back(aux_point);
  pcl::transformPointCloud(*aux_cloud, *result.rotated_pattern, rotation);

  const double zcoord_xyplane = result.rotated_pattern->at(0).z;

  pcl::PointXYZ edges_centroid;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(xy_cloud);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> euclidean_cluster;
  euclidean_cluster.setClusterTolerance(cfg.cluster_tole);
  euclidean_cluster.setMinClusterSize(12);
  euclidean_cluster.setMaxClusterSize(rings_count_v[laser_type] * 4);
  euclidean_cluster.setSearchMethod(tree);
  euclidean_cluster.setInputCloud(xy_cloud);
  euclidean_cluster.extract(cluster_indices);

  for (auto it = cluster_indices.begin(); it < cluster_indices.end(); ++it) {
    float accx = 0.0f, accy = 0.0f, accz = 0.0f;
    for (auto it2 = it->indices.begin(); it2 < it->indices.end(); ++it2) {
      accx += xy_cloud->at(*it2).x;
      accy += xy_cloud->at(*it2).y;
      accz += xy_cloud->at(*it2).z;
    }
    edges_centroid.x = accx / it->indices.size();
    edges_centroid.y = accy / it->indices.size();
    edges_centroid.z = accz / it->indices.size();
  }

  pcl::ModelCoefficients::Ptr coefficients3(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers3(new pcl::PointIndices);
  pcl::SACSegmentation<pcl::PointXYZ> circle_segmentation;
  circle_segmentation.setModelType(pcl::SACMODEL_CIRCLE2D);
  circle_segmentation.setDistanceThreshold(cfg.circle_seg_dis_thre);
  circle_segmentation.setMethodType(pcl::SAC_RANSAC);
  circle_segmentation.setOptimizeCoefficients(true);
  circle_segmentation.setMaxIterations(1000);
  circle_segmentation.setRadiusLimits(cfg.circle_radius - cfg.circle_radius_thre,
                                      cfg.circle_radius + cfg.circle_radius_thre);

  pcl::PointCloud<pcl::PointXYZ>::Ptr copy_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(*xy_cloud, *copy_cloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr circle_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto pt = copy_cloud->points.begin(); pt < copy_cloud->points.end(); ++pt) {
    pt->z = zcoord_xyplane;
  }
  *result.xy_cloud = *copy_cloud;

  pcl::ExtractIndices<pcl::PointXYZ> extract;
  std::vector<std::vector<float> > found_centers;
  std::vector<pcl::PointXYZ> centroid_cloud_inliers;
  bool valid = true;

  while ((copy_cloud->points.size() + centroid_cloud_inliers.size()) > 3 &&
         found_centers.size() < 4 && !copy_cloud->points.empty()) {
    circle_segmentation.setInputCloud(copy_cloud);
    circle_segmentation.segment(*inliers3, *coefficients3);
    if (inliers3->indices.empty()) {
      break;
    }

    extract.setInputCloud(copy_cloud);
    extract.setIndices(inliers3);
    extract.setNegative(false);
    extract.filter(*circle_cloud);
    *result.last_circle_inliers = *circle_cloud;

    pcl::PointXYZ center;
    center.x = *coefficients3->values.begin();
    center.y = *(coefficients3->values.begin() + 1);
    center.z = zcoord_xyplane;
    const double centroid_distance =
        std::sqrt(std::pow(std::fabs(edges_centroid.x - center.x), 2) +
                  std::pow(std::fabs(edges_centroid.y - center.y), 2));
    if (centroid_distance < cfg.centroid_distance_min) {
      valid = false;
      for (auto pt = circle_cloud->points.begin(); pt < circle_cloud->points.end(); ++pt) {
        centroid_cloud_inliers.push_back(*pt);
      }
    } else if (centroid_distance > cfg.centroid_distance_max) {
      valid = false;
    } else {
      for (auto it = found_centers.begin(); it != found_centers.end(); ++it) {
        if (std::sqrt(std::pow(std::fabs((*it)[0] - center.x), 2) +
                      std::pow(std::fabs((*it)[1] - center.y), 2)) < 0.25) {
          valid = false;
          break;
        }
      }
      for (auto pt = centroid_cloud_inliers.begin(); pt < centroid_cloud_inliers.end(); ++pt) {
        const double distance_to_cluster =
            std::sqrt(std::pow(pt->x - center.x, 2) + std::pow(pt->y - center.y, 2) + std::pow(pt->z - center.z, 2));
        if (distance_to_cluster < cfg.circle_radius + 0.02) {
          centroid_cloud_inliers.erase(pt);
          --pt;
        }
      }
    }

    if (valid) {
      found_centers.push_back({center.x, center.y, center.z});
    }

    extract.setNegative(true);
    extract.filter(*cloud_f);
    copy_cloud.swap(cloud_f);
    valid = true;
  }

  result.frame_centers->clear();
  if (found_centers.size() < static_cast<size_t>(cfg.min_centers_found) || found_centers.size() >= 5) {
    return false;
  }

  for (auto it = found_centers.begin(); it < found_centers.end(); ++it) {
    pcl::PointXYZ center;
    center.x = (*it)[0];
    center.y = (*it)[1];
    center.z = (*it)[2];
    pcl::PointXYZ center_rotated_back = pcl::transformPoint(center, rotation.inverse());
    center_rotated_back.x =
        (-coefficients->values[1] * center_rotated_back.y - coefficients->values[2] * center_rotated_back.z - coefficients->values[3]) /
        coefficients->values[0];
    result.frame_centers->push_back(center_rotated_back);
  }

  *result.plane_coefficients = *coefficients;
  return true;
}

inline bool clusterOusterCenters(const pcl::PointCloud<pcl::PointXYZ>::Ptr& frame_centers,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr& cumulative_cloud,
                                 double cluster_size,
                                 int frame_count,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr& centers_cloud) {
  if (!frame_centers || frame_centers->points.empty()) {
    return false;
  }
  if (!cumulative_cloud) {
    cumulative_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
  }
  *cumulative_cloud += *frame_centers;

  if (!centers_cloud) {
    centers_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
  }
  centers_cloud->clear();

  const int safe_frame_count = std::max(1, frame_count);
  getCenterClusters(cumulative_cloud, centers_cloud, cluster_size, std::max(1, safe_frame_count / 2), safe_frame_count, false);
  if (centers_cloud->points.size() > 4) {
    centers_cloud->clear();
    getCenterClusters(cumulative_cloud, centers_cloud, cluster_size, std::max(1, 3 * safe_frame_count / 4), safe_frame_count, false);
  }
  return centers_cloud->points.size() == 4;
}

}  // namespace lvt2calib

#endif
