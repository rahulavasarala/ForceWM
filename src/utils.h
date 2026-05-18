#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <yaml-cpp/yaml.h>

enum class VisualStreamType {
  kRgb,
  kDepth,
};

struct DepthStreamConfig {
  std::string visual_name;
  std::string redis_key;
  std::string metadata_redis_key;
  std::string encoding = "png16";
  std::string unit = "mm";
  std::string align_to_visual_name;
  int width = 640;
  int height = 480;
  int channels = 1;
  std::vector<float> raw_depth_buffer;
  std::vector<float> flipped_depth_buffer;
  std::vector<std::uint16_t> depth_mm_buffer;
  std::vector<unsigned char> encoded_depth_buffer;
};

struct CameraStreamConfig {
  std::string visual_name;
  std::string redis_key;
  std::string metadata_redis_key;
  std::string mujoco_camera_name;
  std::string encoding = "jpeg";
  VisualStreamType type = VisualStreamType::kRgb;
  int model_camera_id = -1;
  int width = 640;
  int height = 480;
  int channels = 3;
  double fps = 0.0;
  mjtNum next_publish_sim_time = 0.0;
  std::vector<unsigned char> rgb_buffer;
  std::vector<unsigned char> flipped_rgb_buffer;
  std::vector<unsigned char> bgr_buffer;
  std::vector<unsigned char> encoded_image_buffer;
  std::optional<DepthStreamConfig> aligned_depth;
  std::uint64_t publish_count = 0;
  std::uint64_t dropped_publish_slots = 0;
  double total_scene_update_seconds = 0.0;
  double total_render_seconds = 0.0;
  double total_render_draw_seconds = 0.0;
  double total_readback_seconds = 0.0;
  double total_flip_seconds = 0.0;
  double total_color_convert_seconds = 0.0;
  double total_frame_encode_seconds = 0.0;
  double total_redis_publish_seconds = 0.0;
  double total_publish_seconds = 0.0;
};

struct VisualContractEntry {
  std::string visual_name;
  VisualStreamType type = VisualStreamType::kRgb;
  std::string source = "sim";
  std::string redis_key;
  std::string metadata_redis_key;
  std::string mujoco_camera_name;
  std::string encoding = "jpeg";
  std::string unit;
  std::string align_to_visual_name;
  int width = 640;
  int height = 480;
  int channels = 3;
  double fps = 0.0;
  bool has_explicit_dimensions = false;
  bool has_explicit_channels = false;
  bool has_explicit_fps = false;
};

struct SimulationContractConfig {
  std::string prefix;
  std::filesystem::path xml_path;
  std::filesystem::path urdf_path;
  std::vector<CameraStreamConfig> cameras;
};

struct StartupOptions {
  std::filesystem::path contract_path;
  bool is_data_collection = false;
};

template <typename T>
T node_or(const YAML::Node& node, const T& fallback) {
  return node ? node.as<T>() : fallback;
}

std::filesystem::path repo_root_directory();
std::filesystem::path default_contract_path();
std::filesystem::path resolve_input_path(const char* argument);
std::filesystem::path resolve_contract_relative_path(
    const std::filesystem::path& contract_path,
    const std::string& path_string);
std::string normalize_mode_token(std::string mode);
bool parse_mode_argument(const std::string& mode_argument,
                         bool& parsed_is_data_collection);
void print_usage(const char* executable_name);
std::optional<StartupOptions> parse_startup_options(int argc, char** argv);
std::string require_string(const YAML::Node& parent,
                           const char* key,
                           const std::string& context);
std::string make_redis_key(const std::string& prefix,
                           const std::string& suffix);
std::string lowercase_copy(std::string value);
VisualStreamType parse_visual_stream_type(const std::string& visual_type,
                                          const std::string& context);
SimulationContractConfig load_simulation_contract(
    const std::filesystem::path& contract_path);
