#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <ros/ros.h>
#include <ros/package.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <lvt2calib/offline_circle_centers.h>

namespace {

struct CliOptions {
  std::string sensor;
  std::string input_path;
  std::string output_path;
  std::string template_path;
  std::string preset;
  int laser_ring_num = -1;

  std::unordered_map<std::string, std::string> overrides;
};

bool parseBool(const std::string& value) {
  if (value == "1" || value == "true" || value == "TRUE" || value == "True") return true;
  if (value == "0" || value == "false" || value == "FALSE" || value == "False") return false;
  throw std::runtime_error("invalid boolean: " + value);
}

double parseDouble(const std::string& value, const std::string& name) {
  try {
    return std::stod(value);
  } catch (...) {
    throw std::runtime_error("invalid value for --" + name + ": " + value);
  }
}

int parseInt(const std::string& value, const std::string& name) {
  try {
    return std::stoi(value);
  } catch (...) {
    throw std::runtime_error("invalid value for --" + name + ": " + value);
  }
}

void printUsage() {
  std::cerr
      << "Usage: pcd_circle_centers --sensor livox|ouster --input <cloud.pcd> --output <centers.json> [options]\n"
      << "Options:\n"
      << "  --template <four_circle_boundary.pcd>\n"
      << "  --preset livox_horizon|livox_mid70|os1_32|os1_64|os1_128\n"
      << "  --laser-ring-num <N>\n"
      << "  --cluster-tole <value> --cluster-size-min <value> --cluster-size-max <value>\n"
      << "  --i-filter-out-max <value> --rmse-ukn2tpl-thre <value> --rmse-tpl2ukn-thre <value>\n"
      << "  --circle-radius <value> --circle-seg-thre <value> --centroid-dis-min <value> --centroid-dis-max <value>\n"
      << "  --cluster-size <value> --circle-radius-thre <value> --circle-seg-dis-thre <value>\n"
      << "  --edge-depth-thre <value> --edge-knn-radius <value> --angle-threshold <value>\n";
}

CliOptions parseArgs(int argc, char** argv) {
  CliOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage();
      std::exit(0);
    }
    if (arg.rfind("--", 0) != 0) {
      throw std::runtime_error("unexpected argument: " + arg);
    }
    if (i + 1 >= argc) {
      throw std::runtime_error("missing value for " + arg);
    }
    const std::string key = arg.substr(2);
    const std::string value = argv[++i];
    if (key == "sensor") {
      options.sensor = value;
    } else if (key == "input") {
      options.input_path = value;
    } else if (key == "output") {
      options.output_path = value;
    } else if (key == "template") {
      options.template_path = value;
    } else if (key == "preset") {
      options.preset = value;
    } else if (key == "laser-ring-num") {
      options.laser_ring_num = parseInt(value, key);
    } else {
      options.overrides[key] = value;
    }
  }
  return options;
}

void applyCommonOverrides(lvt2calib::LaserPatternConfig& cfg,
                          const std::unordered_map<std::string, std::string>& overrides) {
  for (const auto& entry : overrides) {
    const std::string& key = entry.first;
    const std::string& value = entry.second;
    if (key == "remove-x-min") cfg.remove_x_min = parseDouble(value, key);
    else if (key == "remove-x-max") cfg.remove_x_max = parseDouble(value, key);
    else if (key == "use-rg-pseg") cfg.use_RG_Pseg = parseBool(value);
    else if (key == "use-vox-filter") cfg.use_vox_filter = parseBool(value);
    else if (key == "use-i-filter") cfg.use_i_filter = parseBool(value);
    else if (key == "use-gauss-filter") cfg.use_gauss_filter = parseBool(value);
    else if (key == "use-gauss-filter2") cfg.use_gauss_filter2 = parseBool(value);
    else if (key == "use-statistic-filter") cfg.use_statistic_filter = parseBool(value);
    else if (key == "voxel-grid-size") cfg.voxel_grid_size = parseDouble(value, key);
    else if (key == "gauss-k-sigma") cfg.gauss_k_sigma = parseDouble(value, key);
    else if (key == "gauss-k-thre-rt-sigma") cfg.gauss_k_thre_rt_sigma = parseDouble(value, key);
    else if (key == "gauss-k-thre") cfg.gauss_k_thre = parseDouble(value, key);
    else if (key == "gauss-conv-radius") cfg.gauss_conv_radius = parseDouble(value, key);
    else if (key == "cluster-tole") cfg.cluster_tole = parseDouble(value, key);
    else if (key == "cluster-size-min") cfg.cluster_size_min = parseDouble(value, key);
    else if (key == "cluster-size-max") cfg.cluster_size_max = parseDouble(value, key);
    else if (key == "pseg-dis-thre") cfg.Pseg_dis_thre = parseDouble(value, key);
    else if (key == "pseg-iter-num") cfg.Pseg_iter_num = parseInt(value, key);
    else if (key == "pseg-size-min") cfg.Pseg_size_min = parseDouble(value, key);
    else if (key == "rg-smooth-thre-deg") cfg.RG_smooth_thre_deg = parseDouble(value, key);
    else if (key == "rg-curve-thre") cfg.RG_curve_thre = parseDouble(value, key);
    else if (key == "rg-neighbor-n") cfg.RG_neighbor_n = parseInt(value, key);
    else if (key == "sor-mean-k") cfg.sor_MeanK = parseInt(value, key);
    else if (key == "sor-stddev-mul-thresh") cfg.sor_StddevMulThresh = parseInt(value, key);
    else if (key == "i-filter-out-min") cfg.i_filter_out_min = parseDouble(value, key);
    else if (key == "i-filter-out-max") cfg.i_filter_out_max = parseDouble(value, key);
    else if (key == "bound-est-rad") cfg.boundEstRad = parseDouble(value, key);
    else if (key == "norm-est-rad") cfg.normEstRad = parseDouble(value, key);
    else if (key == "rmse-ukn2tpl-thre") cfg.rmse_ukn2tpl_thre = parseDouble(value, key);
    else if (key == "rmse-tpl2ukn-thre") cfg.rmse_tpl2ukn_thre = parseDouble(value, key);
    else if (key == "circle-radius") cfg.circle_radius = parseDouble(value, key);
    else if (key == "circle-seg-thre") cfg.circle_seg_thre = parseDouble(value, key);
    else if (key == "centroid-dis-min") cfg.centroid_dis_min = parseDouble(value, key);
    else if (key == "centroid-dis-max") cfg.centroid_dis_max = parseDouble(value, key);
    else if (key == "min-centers-found") cfg.min_centers_found = parseInt(value, key);
  }
}

void applyOusterCircleOverrides(lvt2calib::OusterCircleConfig& cfg,
                                const std::unordered_map<std::string, std::string>& overrides) {
  for (const auto& entry : overrides) {
    const std::string& key = entry.first;
    const std::string& value = entry.second;
    if (key == "cluster-size") cfg.cluster_size = parseDouble(value, key);
    else if (key == "min-centers-found") cfg.min_centers_found = parseInt(value, key);
    else if (key == "laser-ring-num") cfg.rings_count = parseInt(value, key);
    else if (key == "x") cfg.axis.x() = parseDouble(value, key);
    else if (key == "y") cfg.axis.y() = parseDouble(value, key);
    else if (key == "z") cfg.axis.z() = parseDouble(value, key);
    else if (key == "angle-threshold") cfg.angle_threshold = parseDouble(value, key);
    else if (key == "edge-depth-thre") cfg.edge_depth_thre = parseDouble(value, key);
    else if (key == "edge-knn-radius") cfg.edge_knn_radius = parseDouble(value, key);
    else if (key == "cluster-tole") cfg.cluster_tole = parseDouble(value, key);
    else if (key == "circle-radius") cfg.circle_radius = parseDouble(value, key);
    else if (key == "circle-radius-thre") cfg.circle_radius_thre = parseDouble(value, key);
    else if (key == "circle-seg-dis-thre") cfg.circle_seg_dis_thre = parseDouble(value, key);
    else if (key == "centroid-distance-min") cfg.centroid_distance_min = parseDouble(value, key);
    else if (key == "centroid-distance-max") cfg.centroid_distance_max = parseDouble(value, key);
  }
}

std::string defaultTemplatePath() {
  const std::string pkg_path = ros::package::getPath("lvt2calib");
  if (pkg_path.empty()) {
    return std::string();
  }
  return pkg_path + "/data/template_pcl/four_circle_boundary.pcd";
}

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "pcd_circle_centers", ros::init_options::AnonymousName);

  try {
    const CliOptions options = parseArgs(argc, argv);
    if (options.sensor.empty() || options.input_path.empty() || options.output_path.empty()) {
      printUsage();
      return 1;
    }

    const std::string template_path = options.template_path.empty() ? defaultTemplatePath() : options.template_path;
    if (template_path.empty()) {
      throw std::runtime_error("unable to resolve default template path; pass --template explicitly");
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr template_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile(template_path, *template_cloud) != 0) {
      throw std::runtime_error("failed to load template PCD: " + template_path);
    }

    if (options.sensor == "livox") {
      lvt2calib::LaserPatternConfig config = lvt2calib::makeLivoxPreset(options.preset);
      applyCommonOverrides(config, options.overrides);

      pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZI>);
      if (pcl::io::loadPCDFile(options.input_path, *input_cloud) != 0) {
        throw std::runtime_error("failed to load input PCD: " + options.input_path);
      }

      AutoDetectLaser detector;
      FourCircleCenters four_centers;
      detector.setCalibTemplate(*template_cloud);
      lvt2calib::configureAutoDetectLaser(detector, config, lvt2calib::kLivoxMaxSize);
      lvt2calib::configureFourCircleCenters(four_centers, config);

      pcl::PointCloud<pcl::PointXYZI>::Ptr calib_board(new pcl::PointCloud<pcl::PointXYZI>);
      const bool board_detected = config.use_RG_Pseg ? detector.detectCalibBoardRG(input_cloud, calib_board)
                                                 : detector.detectCalibBoard(input_cloud, calib_board);
      if (!board_detected) {
        throw std::runtime_error("failed to detect Livox calibration board");
      }

      pcl::PointCloud<pcl::PointXYZI>::Ptr centers(new pcl::PointCloud<pcl::PointXYZI>);
      if (!four_centers.FindFourCenters(template_cloud, centers, detector.Tr_calib2tpl_.inverse())) {
        throw std::runtime_error("failed to extract Livox circle centers");
      }

      lvt2calib::CircleJsonOptions json_options;
      json_options.sensor_type = "livox";
      json_options.include_input_pcd = true;
      json_options.input_pcd = options.input_path;
      if (!lvt2calib::writeCircleCentersJson(options.output_path, *centers, json_options)) {
        throw std::runtime_error("failed to write JSON output: " + options.output_path);
      }
      std::cout << "Wrote " << centers->points.size() << " Livox circle centers to " << options.output_path << std::endl;
      return 0;
    }

    if (options.sensor == "ouster") {
      lvt2calib::LaserPatternConfig config = lvt2calib::makeOusterPreset(options.preset);
      lvt2calib::OusterCircleConfig circle_config = lvt2calib::makeOusterCirclePreset(options.preset);
      if (options.laser_ring_num > 0) {
        circle_config.rings_count = options.laser_ring_num;
      }
      applyCommonOverrides(config, options.overrides);
      applyOusterCircleOverrides(circle_config, options.overrides);

      pcl::PointCloud<Ouster::Point>::Ptr input_cloud(new pcl::PointCloud<Ouster::Point>);
      if (pcl::io::loadPCDFile(options.input_path, *input_cloud) != 0) {
        throw std::runtime_error("failed to load input PCD: " + options.input_path);
      }

      findLaserType(circle_config.rings_count);
      Ouster::addRange(*input_cloud);
      Ouster::normalizeIntensity(*input_cloud, 0, 255);
      pcl::PointCloud<pcl::PointXYZI>::Ptr detect_cloud(new pcl::PointCloud<pcl::PointXYZI>);
      pcl::copyPointCloud(*input_cloud, *detect_cloud);
      Ouster::resetIntensity(*input_cloud);

      AutoDetectLaser detector(R_LIDAR);
      detector.setCalibTemplate(*template_cloud);
      lvt2calib::configureAutoDetectLaser(detector, config, lvt2calib::kRepetitiveMaxSize);

      pcl::PointCloud<pcl::PointXYZI>::Ptr calib_board(new pcl::PointCloud<pcl::PointXYZI>);
      const bool board_detected = config.use_RG_Pseg ? detector.detectCalibBoardRG(detect_cloud, calib_board)
                                                 : detector.detectCalibBoard(detect_cloud, calib_board);
      if (!board_detected) {
        throw std::runtime_error("failed to detect Ouster calibration board");
      }

      lvt2calib::OusterCircleExtractionResult extraction;
      if (!lvt2calib::extractOusterCircleFrame(input_cloud, calib_board, circle_config, extraction)) {
        throw std::runtime_error("failed to extract Ouster frame circle centers");
      }

      pcl::PointCloud<pcl::PointXYZ>::Ptr cumulative_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_centers(new pcl::PointCloud<pcl::PointXYZ>);
      if (!lvt2calib::clusterOusterCenters(extraction.frame_centers, cumulative_cloud, circle_config.cluster_size, 1, clustered_centers)) {
        if (extraction.frame_centers->points.size() == 4) {
          clustered_centers = extraction.frame_centers;
        } else {
          throw std::runtime_error("failed to cluster Ouster circle centers");
        }
      }

      lvt2calib::CircleJsonOptions json_options;
      json_options.sensor_type = "ouster";
      json_options.include_input_pcd = true;
      json_options.input_pcd = options.input_path;
      if (!lvt2calib::writeCircleCentersJson(options.output_path, *clustered_centers, json_options)) {
        throw std::runtime_error("failed to write JSON output: " + options.output_path);
      }
      std::cout << "Wrote " << clustered_centers->points.size() << " Ouster circle centers to " << options.output_path << std::endl;
      return 0;
    }

    throw std::runtime_error("unsupported sensor: " + options.sensor);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  }
}
