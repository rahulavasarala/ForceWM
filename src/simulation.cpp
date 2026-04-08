#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "SaiModel.h"
#include "SaiPrimitives.h"

#include "redis/RedisClient.h"
#include "redis_keys.h"
#include "redis/keys/chai_haptic_devices_driver.h"

SaiCommon::RedisClient redis_client = SaiCommon::RedisClient("sai");;

using std::string;
using namespace SaiCommon::ChaiHapticDriverKeys;
namespace fs = std::filesystem;

namespace {

constexpr int kWindowWidth = 1200;
constexpr int kWindowHeight = 900;
constexpr int kSceneMaxGeometry = 2000;
constexpr mjtNum kRenderTimestep = 1.0 / 60.0;
constexpr int kRobotDof = 7;
constexpr const char* kDefaultSceneXmlPath = "models/scene.xml";
constexpr const char* kDefaultRobotUrdfPath = "models/fr3.urdf";
constexpr const char* kEndEffectorForceSensorName = "ee_force";
constexpr const char* kEndEffectorTorqueSensorName = "ee_torque";
constexpr const char* kEndEffectorSensorSiteName = "ft_site";

mjModel* m = nullptr;
mjData* d = nullptr;
mjData* viewer_data = nullptr;
mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0.0;
double lasty = 0.0;

// ---- Mode -----
bool is_data_collection = false; 
// ---------------

// Initialization variables for robot control ---------------------

Vector3d START_POS = Vector3d(0.4, 0.0, 0.1);
Matrix3d START_ORIENTATION = (Matrix3d() << 
    1,  0,  0,
    0, -1,  0,
    0,  0, -1).finished();

std::shared_ptr<SaiModel::SaiModel> robot;
std::shared_ptr<SaiPrimitives::MotionForceTask> motion_force_task;
std::shared_ptr<SaiPrimitives::JointTask> joint_task;

std::shared_ptr<SaiPrimitives::HapticDeviceController> haptic_controller;
SaiPrimitives::HapticControllerInput haptic_input;
SaiPrimitives::HapticControllerOutput haptic_output;

Vector3d prev_sensed_force;
Vector3i directions_of_proxy_feedback;

Vector3d control_point;
Affine3d control_frame;

std::string control_link = "fr3_link8";
int ee_force_sensor_id = -1;
int ee_torque_sensor_id = -1;
int ee_sensor_site_id = -1;
std::atomic<bool> shutdown_requested = false;
std::atomic<bool> reset_requested = false;

struct CameraStreamConfig {
  std::string redis_key;
  std::string metadata_redis_key;
  std::string mujoco_camera_name;
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
  std::uint64_t publish_count = 0;
  std::uint64_t dropped_publish_slots = 0;
  double total_render_seconds = 0.0;
  double total_readback_seconds = 0.0;
  double total_jpeg_encode_seconds = 0.0;
  double total_redis_publish_seconds = 0.0;
  double total_publish_seconds = 0.0;
};

struct SimulationContractConfig {
  std::string prefix;
  fs::path xml_path;
  fs::path urdf_path;
  std::vector<CameraStreamConfig> cameras;
};

struct StartupOptions {
  fs::path contract_path;
  bool is_data_collection = false;
};

struct SimulationPerformanceStats {
  std::uint64_t physics_step_count = 0;
  std::uint64_t rendered_frame_count = 0;
  mjtNum simulated_seconds = 0.0;
  double wall_seconds = 0.0;
};

struct SimulationLoopStats {
  std::uint64_t physics_step_count = 0;
  mjtNum sim_start_time = 0.0;
  mjtNum sim_end_time = 0.0;
};

struct RenderSnapshot {
  std::uint64_t seq = 0;
  std::uint64_t reset_epoch = 0;
  mjtNum sim_time = 0.0;
  double publish_wall_time_s = 0.0;
  std::vector<mjtNum> qpos;
  std::vector<mjtNum> qvel;
  std::vector<mjtNum> act;
  std::vector<mjtNum> mocap_pos;
  std::vector<mjtNum> mocap_quat;
  std::vector<mjtNum> userdata;
};

struct SnapshotBroker {
  explicit SnapshotBroker(const mjModel* model) {
    latest_snapshot.qpos.resize(model->nq);
    latest_snapshot.qvel.resize(model->nv);
    latest_snapshot.act.resize(model->na);
    latest_snapshot.mocap_pos.resize(3 * model->nmocap);
    latest_snapshot.mocap_quat.resize(4 * model->nmocap);
    latest_snapshot.userdata.resize(model->nuserdata);
  }

  void publish_from_sim(const mjData* source,
                        const std::uint64_t reset_epoch,
                        const double publish_wall_time_s) {
    std::lock_guard<std::mutex> lock(mutex);
    ++latest_snapshot.seq;
    latest_snapshot.reset_epoch = reset_epoch;
    latest_snapshot.sim_time = source->time;
    latest_snapshot.publish_wall_time_s = publish_wall_time_s;

    if (m->nq > 0) {
      std::copy_n(source->qpos, m->nq, latest_snapshot.qpos.data());
    }
    if (m->nv > 0) {
      std::copy_n(source->qvel, m->nv, latest_snapshot.qvel.data());
    }
    if (m->na > 0) {
      std::copy_n(source->act, m->na, latest_snapshot.act.data());
    }
    if (m->nmocap > 0) {
      std::copy_n(source->mocap_pos, 3 * m->nmocap,
                  latest_snapshot.mocap_pos.data());
      std::copy_n(source->mocap_quat, 4 * m->nmocap,
                  latest_snapshot.mocap_quat.data());
    }
    if (m->nuserdata > 0) {
      std::copy_n(source->userdata, m->nuserdata,
                  latest_snapshot.userdata.data());
    }

    has_snapshot = true;
    condition.notify_all();
  }

  bool copy_latest(RenderSnapshot& out_snapshot) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (!has_snapshot) {
      return false;
    }

    copy_snapshot_locked(latest_snapshot, out_snapshot);
    return true;
  }

  bool wait_for_newer(const std::uint64_t last_seq,
                      RenderSnapshot& out_snapshot,
                      const std::atomic<bool>& stop_flag) const {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] {
      return stop_flag.load() ||
             (has_snapshot && latest_snapshot.seq > last_seq);
    });

    if (stop_flag.load()) {
      return false;
    }

    copy_snapshot_locked(latest_snapshot, out_snapshot);
    return true;
  }

  void notify_all() const {
    condition.notify_all();
  }

 private:
  static void copy_snapshot_locked(const RenderSnapshot& source,
                                   RenderSnapshot& destination) {
    destination.seq = source.seq;
    destination.reset_epoch = source.reset_epoch;
    destination.sim_time = source.sim_time;
    destination.publish_wall_time_s = source.publish_wall_time_s;
    destination.qpos = source.qpos;
    destination.qvel = source.qvel;
    destination.act = source.act;
    destination.mocap_pos = source.mocap_pos;
    destination.mocap_quat = source.mocap_quat;
    destination.userdata = source.userdata;
  }

  mutable std::mutex mutex;
  mutable std::condition_variable condition;
  RenderSnapshot latest_snapshot;
  bool has_snapshot = false;
};

SimulationContractConfig simulation_contract;
std::unique_ptr<SnapshotBroker> snapshot_broker;

// Initialization variables for robot control ---------------------

std::optional<fs::path> plugin_directory() {
  if (const char* plugin_path = std::getenv("MUJOCO_PLUGIN_PATH")) {
    const fs::path path(plugin_path);
    if (fs::exists(path)) {
      return path;
    }
  }

  if (const char* conda_prefix = std::getenv("CONDA_PREFIX")) {
    const fs::path path = fs::path(conda_prefix) / "bin" / "mujoco_plugin";
    if (fs::exists(path)) {
      return path;
    }
  }

  return std::nullopt;
}

void load_mujoco_plugins() {
  const auto path = plugin_directory();
  if (!path) {
    std::cout << "No MuJoCo plugin directory found. Continuing without extra "
                 "plugins.\n";
    return;
  }

  std::cout << "Loading MuJoCo plugins from: " << path->string() << "\n";
  mj_loadAllPluginLibraries(path->string().c_str(), nullptr);
}

fs::path repo_root_directory() {
  return fs::path(FORCEWM_MODEL_ROOT).parent_path();
}

fs::path default_contract_path() {
  return repo_root_directory() / "universal_contract.yaml";
}

fs::path resolve_input_path(const char* argument) {
  if (!argument) {
    return default_contract_path();
  }

  const fs::path input_path(argument);
  if (input_path.is_absolute()) {
    return input_path.lexically_normal();
  }

  return fs::absolute(input_path).lexically_normal();
}

fs::path resolve_contract_relative_path(const fs::path& contract_path,
                                        const std::string& path_string) {
  const fs::path input_path(path_string);
  if (input_path.is_absolute()) {
    return input_path;
  }
  return (contract_path.parent_path() / input_path).lexically_normal();
}

std::string normalize_mode_token(std::string mode) {
  std::transform(mode.begin(), mode.end(), mode.begin(),
                 [](unsigned char character) {
                   if (character == '-' || character == ' ') {
                     return static_cast<char>('_');
                   }
                   return static_cast<char>(std::tolower(character));
                 });
  return mode;
}

bool parse_mode_argument(const std::string& mode_argument,
                         bool& parsed_is_data_collection) {
  const std::string normalized_mode = normalize_mode_token(mode_argument);
  if (normalized_mode == "inference") {
    parsed_is_data_collection = false;
    return true;
  }

  if (normalized_mode == "data_collection" ||
      normalized_mode == "datacollection" ||
      normalized_mode == "collection") {
    parsed_is_data_collection = true;
    return true;
  }

  return false;
}

void print_usage(const char* executable_name) {
  std::cout << "Usage: " << executable_name
            << " [universal_contract.yaml] [inference|data_collection]\n"
            << "Defaults are loaded from " << default_contract_path().string()
            << " and mode defaults to inference.\n"
            << "\n"
            << "Examples:\n"
            << "  " << executable_name << "\n"
            << "  " << executable_name << " data_collection\n"
            << "  " << executable_name
            << " /full/path/to/universal_contract.yaml inference\n";
}

void print_simulation_summary(const SimulationPerformanceStats& stats) {
  std::cout << "\nSimulation summary\n";
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  Wall time: " << stats.wall_seconds << " s\n";
  std::cout << "  Simulated time: " << stats.simulated_seconds << " s\n";
  std::cout << "  Physics steps: " << stats.physics_step_count << "\n";
  std::cout << "  Rendered frames: " << stats.rendered_frame_count << "\n";

  if (stats.wall_seconds > 1e-9) {
    std::cout << "  Real-time factor: "
              << static_cast<double>(stats.simulated_seconds) /
                     stats.wall_seconds
              << "x\n";
    std::cout << "  Physics step rate: "
              << static_cast<double>(stats.physics_step_count) /
                     stats.wall_seconds
              << " steps/s\n";
    std::cout << "  Render FPS: "
              << static_cast<double>(stats.rendered_frame_count) /
                     stats.wall_seconds
              << " frames/s\n";
  } else {
    std::cout << "  Runtime too short to compute rates.\n";
  }

  if (stats.rendered_frame_count > 0) {
    std::cout << "  Avg steps per frame: "
              << static_cast<double>(stats.physics_step_count) /
                     static_cast<double>(stats.rendered_frame_count)
              << "\n";
  }
  std::cout << std::defaultfloat;
}

void print_camera_publish_summary() {
  if (simulation_contract.cameras.empty()) {
    return;
  }

  std::cout << "\nCamera publish summary\n";
  std::cout << std::fixed << std::setprecision(3);

  for (const auto& camera : simulation_contract.cameras) {
    std::cout << "  Camera `" << camera.mujoco_camera_name << "` -> `"
              << camera.redis_key << "`\n";

    if (camera.publish_count == 0) {
      std::cout << "    No images published.\n";
      continue;
    }

    const double publish_count = static_cast<double>(camera.publish_count);
    const double avg_render_ms =
        1000.0 * camera.total_render_seconds / publish_count;
    const double avg_readback_ms =
        1000.0 * camera.total_readback_seconds / publish_count;
    const double avg_jpeg_ms =
        1000.0 * camera.total_jpeg_encode_seconds / publish_count;
    const double avg_redis_ms =
        1000.0 * camera.total_redis_publish_seconds / publish_count;
    const double avg_total_ms =
        1000.0 * camera.total_publish_seconds / publish_count;

    std::cout << "    Published frames: " << camera.publish_count << "\n";
    std::cout << "    Dropped publish slots: " << camera.dropped_publish_slots
              << "\n";
    std::cout << "    Avg render time: " << avg_render_ms << " ms\n";
    std::cout << "    Avg readback time: " << avg_readback_ms << " ms\n";
    std::cout << "    Avg JPEG encode time: " << avg_jpeg_ms << " ms\n";
    std::cout << "    Avg Redis publish time: " << avg_redis_ms << " ms\n";
    std::cout << "    Avg total publish time: " << avg_total_ms << " ms\n";
  }

  std::cout << std::defaultfloat;
}

std::optional<StartupOptions> parse_startup_options(int argc, char** argv) {
  if (argc > 3) {
    print_usage(argv[0]);
    return std::nullopt;
  }

  StartupOptions options;
  const char* contract_argument = nullptr;
  const char* mode_argument = nullptr;

  if (argc == 2) {
    if (!parse_mode_argument(argv[1], options.is_data_collection)) {
      contract_argument = argv[1];
    }
  } else if (argc == 3) {
    contract_argument = argv[1];
    mode_argument = argv[2];
  }

  if (mode_argument &&
      !parse_mode_argument(mode_argument, options.is_data_collection)) {
    std::cerr << "Invalid mode `" << mode_argument
              << "`. Expected `inference` or `data_collection`.\n";
    print_usage(argv[0]);
    return std::nullopt;
  }

  options.contract_path = resolve_input_path(contract_argument);
  return options;
}

template <typename T>
T node_or(const YAML::Node& node, const T& fallback) {
  return node ? node.as<T>() : fallback;
}

std::string require_string(const YAML::Node& parent,
                           const char* key,
                           const std::string& context) {
  const YAML::Node value = parent[key];
  if (!value) {
    throw std::runtime_error("Missing required `" + std::string(key) +
                             "` in " + context + ".");
  }

  const std::string parsed_value = value.as<std::string>();
  if (parsed_value.empty()) {
    throw std::runtime_error("Required `" + std::string(key) + "` in " +
                             context + " cannot be empty.");
  }

  return parsed_value;
}

std::string make_redis_key(const std::string& prefix,
                           const std::string& suffix) {
  if (prefix.empty()) {
    return suffix;
  }
  return prefix + "::" + suffix;
}

double wall_time_now_seconds() {
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

void flip_rgb_image_vertically(const std::vector<unsigned char>& source,
                               std::vector<unsigned char>& destination,
                               int width,
                               int height,
                               int channels) {
  const size_t row_stride = static_cast<size_t>(width) * channels;
  destination.resize(source.size());

  for (int row = 0; row < height; ++row) {
    const size_t src_offset =
        static_cast<size_t>(height - 1 - row) * row_stride;
    const size_t dst_offset = static_cast<size_t>(row) * row_stride;
    std::copy_n(source.data() + src_offset, row_stride,
                destination.data() + dst_offset);
  }
}

void copy_snapshot_into_data(const RenderSnapshot& snapshot, mjData* target) {
  target->time = snapshot.sim_time;

  if (m->nq > 0 && !snapshot.qpos.empty()) {
    std::copy_n(snapshot.qpos.data(), m->nq, target->qpos);
  }
  if (m->nv > 0 && !snapshot.qvel.empty()) {
    std::copy_n(snapshot.qvel.data(), m->nv, target->qvel);
  }
  if (m->na > 0 && !snapshot.act.empty()) {
    std::copy_n(snapshot.act.data(), m->na, target->act);
  }
  if (m->nmocap > 0 && !snapshot.mocap_pos.empty() &&
      !snapshot.mocap_quat.empty()) {
    std::copy_n(snapshot.mocap_pos.data(), 3 * m->nmocap, target->mocap_pos);
    std::copy_n(snapshot.mocap_quat.data(), 4 * m->nmocap, target->mocap_quat);
  }
  if (m->nuserdata > 0 && !snapshot.userdata.empty()) {
    std::copy_n(snapshot.userdata.data(), m->nuserdata, target->userdata);
  }
}

void rebuild_render_data(const mjModel* model, mjData* target) {
  mj_fwdPosition(model, target);
}

void publish_snapshot_from_sim_state(const std::uint64_t reset_epoch) {
  if (!snapshot_broker) {
    return;
  }

  snapshot_broker->publish_from_sim(d, reset_epoch, wall_time_now_seconds());
}

std::string make_camera_metadata_json(const CameraStreamConfig& camera,
                                      const RenderSnapshot& snapshot) {
  std::ostringstream metadata_stream;
  metadata_stream << std::fixed << std::setprecision(17);
  metadata_stream << '{';
  metadata_stream << '"' << "seq" << '"' << ':' << snapshot.seq;
  metadata_stream << ',' << '"' << "reset_epoch" << '"' << ':'
                  << snapshot.reset_epoch;
  metadata_stream << ',' << '"' << "camera_name" << '"' << ':' << '"'
                  << camera.mujoco_camera_name << '"';
  metadata_stream << ',' << '"' << "sim_time_s" << '"' << ':'
                  << snapshot.sim_time;
  metadata_stream << ',' << '"' << "publish_wall_time_s" << '"' << ':'
                  << snapshot.publish_wall_time_s;
  metadata_stream << ',' << '"' << "width" << '"' << ':' << camera.width;
  metadata_stream << ',' << '"' << "height" << '"' << ':'
                  << camera.height;
  metadata_stream << ',' << '"' << "channels" << '"' << ':'
                  << camera.channels;
  metadata_stream << ',' << '"' << "encoding" << '"' << ':' << '"'
                  << "jpeg" << '"';
  metadata_stream << ',' << '"' << "jpeg_quality" << '"' << ':' << 90;
  metadata_stream << ',' << '"' << "dropped_slots_total" << '"' << ':'
                  << camera.dropped_publish_slots;
  metadata_stream << '}';
  return metadata_stream.str();
}

SimulationContractConfig load_simulation_contract(const fs::path& contract_path) {
  const YAML::Node contract = YAML::LoadFile(contract_path.string());
  const YAML::Node robot_cfg = contract["robot"];
  if (!robot_cfg || !robot_cfg.IsMap()) {
    throw std::runtime_error("Expected a top-level `robot` mapping in " +
                             contract_path.string() + ".");
  }

  const std::string contract_context =
      "contract `" + contract_path.string() + "`";
  const std::string robot_type =
      require_string(robot_cfg, "type", contract_context);
  if (robot_type != "sim") {
    throw std::runtime_error(
        "simulation.cpp only supports `robot.type: sim`, but the contract sets "
        "`robot.type: " +
        robot_type + "`.");
  }

  SimulationContractConfig config;
  config.prefix = node_or<std::string>(robot_cfg["prefix"], "");
  config.xml_path = resolve_contract_relative_path(
      contract_path,
      node_or<std::string>(robot_cfg["xml_path"], kDefaultSceneXmlPath));
  config.urdf_path = resolve_contract_relative_path(
      contract_path,
      node_or<std::string>(robot_cfg["urdf_path"], kDefaultRobotUrdfPath));

  const YAML::Node visual_cfg =
      robot_cfg["data_sources"] ? robot_cfg["data_sources"]["visual"]
                                : YAML::Node();
  const double default_camera_fps =
      node_or<double>(visual_cfg["fps"], 1.0 / kRenderTimestep);
  const YAML::Node camera_keys = visual_cfg ? visual_cfg["keys"] : YAML::Node();

  if (camera_keys && !camera_keys.IsSequence()) {
    throw std::runtime_error(
        "`robot.data_sources.visual.keys` must be a sequence in " +
        contract_path.string() + ".");
  }

  if (!camera_keys) {
    return config;
  }

  for (const auto& camera_entry : camera_keys) {
    if (!camera_entry.IsMap() || camera_entry.size() != 1) {
      throw std::runtime_error(
          "Each entry in `robot.data_sources.visual.keys` must be a single-key "
          "mapping in " +
          contract_path.string() + ".");
    }

    const auto camera_it = camera_entry.begin();
    const std::string visual_name = camera_it->first.as<std::string>();
    const YAML::Node camera_cfg = camera_it->second;
    const std::string camera_context =
        "camera `" + visual_name + "` in " + contract_context;

    CameraStreamConfig camera;
    const std::string redis_suffix =
        node_or<std::string>(camera_cfg["redis"], visual_name);
    camera.redis_key = make_redis_key(config.prefix, redis_suffix);
    camera.metadata_redis_key = camera.redis_key + "::meta";
    camera.mujoco_camera_name =
        require_string(camera_cfg, "mujoco_camera_name", camera_context);
    camera.fps = node_or<double>(camera_cfg["fps"], default_camera_fps);

    const YAML::Node dim_cfg = camera_cfg["dim"];
    if (dim_cfg && dim_cfg.IsSequence()) {
      if (dim_cfg.size() >= 2) {
        camera.width = dim_cfg[0].as<int>();
        camera.height = dim_cfg[1].as<int>();
      }
      if (dim_cfg.size() >= 3) {
        camera.channels = dim_cfg[2].as<int>();
      }
    }

    if (camera.channels != 3) {
      throw std::runtime_error("Camera '" + visual_name + "' requests " +
                               std::to_string(camera.channels) +
                               " channels, but simulation publishing only "
                               "supports RGB cameras right now.");
    }
    if (camera.fps <= 0.0) {
      throw std::runtime_error("Camera '" + visual_name +
                               "' has a non-positive fps in " +
                               contract_path.string() + ".");
    }

    const size_t image_size =
        static_cast<size_t>(camera.width) * camera.height * camera.channels;
    camera.rgb_buffer.resize(image_size);
    camera.flipped_rgb_buffer.resize(image_size);
    camera.bgr_buffer.resize(image_size);
    config.cameras.push_back(std::move(camera));
  }

  return config;
}

void preflight_camera_publishers() {
  if (simulation_contract.cameras.empty()) {
    std::cout << "No visual keys found in the contract. Camera publishing is "
                 "disabled.\n";
    return;
  }

  int required_offscreen_width = m->vis.global.offwidth;
  int required_offscreen_height = m->vis.global.offheight;

  for (auto& camera : simulation_contract.cameras) {
    camera.model_camera_id =
        mj_name2id(m, mjOBJ_CAMERA, camera.mujoco_camera_name.c_str());
    if (camera.model_camera_id < 0) {
      throw std::runtime_error("MuJoCo camera '" + camera.mujoco_camera_name +
                               "' was not found in " +
                               simulation_contract.xml_path.string() + ".");
    }

    required_offscreen_width = std::max(required_offscreen_width, camera.width);
    required_offscreen_height =
        std::max(required_offscreen_height, camera.height);

    std::cout << "Camera publisher ready: MuJoCo camera '"
              << camera.mujoco_camera_name << "' -> Redis key '"
              << camera.redis_key << "' at " << camera.fps << " Hz ("
              << camera.width << "x" << camera.height << ").\n";
  }

  m->vis.global.offwidth = required_offscreen_width;
  m->vis.global.offheight = required_offscreen_height;
}

void reset_to_home() {
  const int home_keyframe_id = mj_name2id(m, mjOBJ_KEY, "home");
  if (home_keyframe_id >= 0) {
    mj_resetDataKeyframe(m, d, home_keyframe_id);
  } else {
    mj_resetData(m, d);
  }

  if (m->nu > 0) {
    std::fill(d->ctrl, d->ctrl + m->nu, 0.0);
  }

  mj_forward(m, d);
}

void keyboard(GLFWwindow* window, int key, int, int act, int) {
  if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
    reset_requested = true;
  } else if (act == GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
    shutdown_requested = true;
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
}

void mouse_button(GLFWwindow* window, int, int, int) {
  button_left =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
  button_middle =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
  button_right =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

  glfwGetCursorPos(window, &lastx, &lasty);
}

void mouse_move(GLFWwindow* window, double xpos, double ypos) {
  if (!button_left && !button_middle && !button_right) {
    return;
  }

  const double dx = xpos - lastx;
  const double dy = ypos - lasty;
  lastx = xpos;
  lasty = ypos;

  int width = 0;
  int height = 0;
  glfwGetWindowSize(window, &width, &height);
  if (height <= 0) {
    return;
  }

  const bool mod_shift =
      (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
       glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

  mjtMouse action;
  if (button_right) {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (button_left) {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }

  mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

void scroll(GLFWwindow*, double, double yoffset) {
  mjv_moveCamera(m, mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, &scn, &cam);
}

void initialize_camera() {
  mjv_defaultCamera(&cam);
  cam.type = mjCAMERA_FREE;
  cam.azimuth = 135.0;
  cam.elevation = -20.0;
  cam.distance = 2.2;
  cam.lookat[0] = 0.0;
  cam.lookat[1] = 0.0;
  cam.lookat[2] = 0.6;
}

}  // namespace

void controller_callback(const mjModel* m, mjData* d);
void update_robot_state(const mjModel* m, const mjData* d);
void update_redis(const mjModel* m, mjData* d);
void query_redis_for_desired_state();
void update_haptic_information(std::shared_ptr<SaiModel::SaiModel> robot);
void send_haptic_commands();

bool initialize_force_sensor_handles(const mjModel* model) {
  ee_force_sensor_id =
      mj_name2id(model, mjOBJ_SENSOR, kEndEffectorForceSensorName);
  ee_torque_sensor_id =
      mj_name2id(model, mjOBJ_SENSOR, kEndEffectorTorqueSensorName);
  ee_sensor_site_id =
      mj_name2id(model, mjOBJ_SITE, kEndEffectorSensorSiteName);

  if (ee_force_sensor_id < 0 || ee_torque_sensor_id < 0 ||
      ee_sensor_site_id < 0) {
    std::cerr << "Could not find MuJoCo force/torque sensors `"
              << kEndEffectorForceSensorName << "` and `"
              << kEndEffectorTorqueSensorName << "` or site `"
              << kEndEffectorSensorSiteName << "` in the loaded model.\n";
    return false;
  }

  if (model->sensor_dim[ee_force_sensor_id] < 3 ||
      model->sensor_dim[ee_torque_sensor_id] < 3) {
    std::cerr << "MuJoCo force/torque sensors must each provide 3 values.\n";
    return false;
  }

  return true;
}

Matrix3d get_force_sensor_rotation_in_world(const mjData* d) {
  if (!d || ee_sensor_site_id < 0) {
    return Matrix3d::Identity();
  }

  const mjtNum* site_rotation = d->site_xmat + 9 * ee_sensor_site_id;
  Matrix3d world_rotation;
  world_rotation << site_rotation[0], site_rotation[1], site_rotation[2],
      site_rotation[3], site_rotation[4], site_rotation[5], site_rotation[6],
      site_rotation[7], site_rotation[8];
  return world_rotation;
}

void update_robot_state(const mjModel* m, const mjData* d) {
  VectorXd robot_q(kRobotDof);
  VectorXd robot_dq(kRobotDof);

  for (int i = 0; i < kRobotDof; ++i) {
    robot_q(i) = d->qpos[i];
    robot_dq(i) = d->qvel[i];
  }

  robot->setQ(robot_q);
  robot->setDq(robot_dq);
  robot->updateModel();
}

Vector3d get_sensed_force(const mjModel* m, mjData* d,
                         const bool express_in_world_frame = false) {
  if (!m || !d || ee_force_sensor_id < 0) {
    return Vector3d::Zero();
  }

  const int sensor_address = m->sensor_adr[ee_force_sensor_id];
  const Vector3d sensed_force_sensor_frame(
      d->sensordata[sensor_address + 0], d->sensordata[sensor_address + 1],
      d->sensordata[sensor_address + 2]);
  if (!express_in_world_frame) {
    return sensed_force_sensor_frame;
  }

  return get_force_sensor_rotation_in_world(d) * sensed_force_sensor_frame;
}

Vector3d get_sensed_moment(const mjModel* m, mjData* d,
                          const bool express_in_world_frame = false) {
  if (!m || !d || ee_torque_sensor_id < 0) {
    return Vector3d::Zero();
  }

  const int sensor_address = m->sensor_adr[ee_torque_sensor_id];
  const Vector3d sensed_moment_sensor_frame(
      d->sensordata[sensor_address + 0], d->sensordata[sensor_address + 1],
      d->sensordata[sensor_address + 2]);
  if (!express_in_world_frame) {
    return sensed_moment_sensor_frame;
  }

  return get_force_sensor_rotation_in_world(d) * sensed_moment_sensor_frame;
}

void inference_time_callback(const mjModel* m, mjData* d) {

  //When the robot is in inference mode, we have things that are ---- 
  // only relevant to inference and not data collection.

  update_robot_state(m, d);
  update_redis(m, d);
  query_redis_for_desired_state();

  motion_force_task->updateTaskModel(MatrixXd::Identity(robot->dof(), robot->dof()));
  joint_task->updateTaskModel(motion_force_task->getTaskAndPreviousNullspace());
  const VectorXd control_torques =
      motion_force_task->computeTorques() + joint_task->computeTorques() +
      robot->jointGravityVector();

  // const VectorXd control_torques = robot->jointGravityVector();

  // const VectorXd control_torques =
  //     motion_force_task->computeTorques() + joint_task->computeTorques();

  for (int i = 0; i < std::min<int>(control_torques.size(), m->nu); ++i) {
    d->ctrl[i] = control_torques(i);
  }

}

void data_collection_time_callback(const mjModel* m, mjData* d) {

  //If inference mode is false ---- then we are in data collection mo

  update_robot_state(m, d);
  update_redis(m, d);
  update_haptic_information(robot);
  haptic_output = haptic_controller->computeHapticControl(haptic_input);

  send_haptic_commands();

  motion_force_task->updateSensedForceAndMoment(
     -1 * get_sensed_force(m, d),
      -1 *get_sensed_moment(m, d));

  motion_force_task->setGoalPosition(haptic_output.robot_goal_position);
  motion_force_task->setGoalOrientation(
    haptic_output.robot_goal_orientation);

  motion_force_task->updateTaskModel(MatrixXd::Identity(robot->dof(), robot->dof()));
  joint_task->updateTaskModel(motion_force_task->getTaskAndPreviousNullspace());

  const VectorXd control_torques =
      motion_force_task->computeTorques() + joint_task->computeTorques() +
      robot->jointGravityVector();

  Vector3d sensed_force_world_frame = -1 *get_sensed_force(m, d, true);

  for (int i = 0; i < 3; ++i) {
    if (fabs(sensed_force_world_frame(i)) >= 0.5 &&
        fabs(prev_sensed_force(i)) < 0.5) {
      directions_of_proxy_feedback(i) = 1;
    } else if (fabs(sensed_force_world_frame(i)) <= 0.1 &&
               fabs(prev_sensed_force(i)) > 0.1) {
      directions_of_proxy_feedback(i) = 0;
    }
  }

  prev_sensed_force = sensed_force_world_frame;

  const int dim_proxy_space = directions_of_proxy_feedback.sum();
  switch (dim_proxy_space) {
    case 0:
      haptic_controller->parametrizeProxyForceFeedbackSpace(0);
      break;
    case 1:
      haptic_controller->parametrizeProxyForceFeedbackSpace(
          1, directions_of_proxy_feedback.cast<double>());
      break;
    case 2:
      haptic_controller->parametrizeProxyForceFeedbackSpace(
          2, Vector3d::Ones() - directions_of_proxy_feedback.cast<double>());
      break;
    case 3:
      haptic_controller->parametrizeProxyForceFeedbackSpace(3,
                                                            Vector3d::Zero());
      break;
    default:
      break;
  }

  for (int i = 0; i < std::min<int>(control_torques.size(), m->nu); ++i) {
    d->ctrl[i] = control_torques(i);
  }
}


void controller_callback(const mjModel* m, mjData* d) {
  if (!robot || !motion_force_task || !joint_task) {
    return;
  }

  if (is_data_collection) {
    data_collection_time_callback(m, d);
  } else {
    inference_time_callback(m, d);
  }
}

// -------- Redis Code ---------------------------------

void init_redis() {
    redis_client.setEigen(DESIRED_CARTESIAN_POSITION, START_POS);
    redis_client.setEigen(DESIRED_CARTESIAN_ORIENTATION, START_ORIENTATION);
    redis_client.setBool(RESET, false);
}

void update_redis(const mjModel* m, mjData* d) {
    Vector3d currentPosition = motion_force_task->getCurrentPosition();
    Matrix3d currentOrientation = motion_force_task->getCurrentOrientation();
    Vector3d currentLinearVelocity =
        motion_force_task->getCurrentLinearVelocity();

    redis_client.setEigen(CURRENT_CARTESIAN_POSITION, currentPosition);
    redis_client.setEigen(CURRENT_CARTESIAN_ORIENTATION, currentOrientation);
    redis_client.setEigen(CURRENT_CARTESIAN_VELOCITY, currentLinearVelocity);

    VectorXd qpos(kRobotDof);
    for (int i = 0; i < kRobotDof; ++i) {
      qpos(i) = d->qpos[i];
    }

    redis_client.setEigen(QPOS, qpos);

    Vector3d sensed_force = -1 * get_sensed_force(m, d, true);
    Vector3d sensed_moment = -1 *get_sensed_moment(m, d, true);
    redis_client.setEigen(SENSED_FORCE, sensed_force);
    redis_client.setEigen(SENSED_MOMENT, sensed_moment);
}

void query_redis_for_desired_state() {
    const MatrixXd desired_position = redis_client.getEigen(DESIRED_CARTESIAN_POSITION);
    const MatrixXd desired_orientation = redis_client.getEigen(DESIRED_CARTESIAN_ORIENTATION);

    motion_force_task->setGoalPosition(desired_position.col(0).head<3>());
    motion_force_task->setGoalOrientation(desired_orientation.topLeftCorner<3, 3>());
}

void update_haptic_information(std::shared_ptr<SaiModel::SaiModel> robot) {
  haptic_input.device_position =
      redis_client.getEigen(createRedisKey(POSITION_KEY_SUFFIX, 0));
  haptic_input.device_orientation =
      redis_client.getEigen(createRedisKey(ROTATION_KEY_SUFFIX, 0));
  haptic_input.device_linear_velocity =
      redis_client.getEigen(createRedisKey(LINEAR_VELOCITY_KEY_SUFFIX, 0));
  haptic_input.device_angular_velocity =
      redis_client.getEigen(createRedisKey(ANGULAR_VELOCITY_KEY_SUFFIX, 0));

  haptic_input.robot_position = robot->positionInWorld(control_link);
  haptic_input.robot_orientation = robot->rotationInWorld(control_link);
  haptic_input.robot_linear_velocity =
    robot->linearVelocityInWorld(control_link);
  haptic_input.robot_angular_velocity =
    robot->angularVelocityInWorld(control_link);
  // This example still uses proxy feedback instead of direct wrench feedback.
  // The MuJoCo helpers above can now return either sensor-frame or world-frame
  // wrench depending on the call-site flag.
  haptic_input.robot_sensed_force = -1 *get_sensed_force(m, d, true);
  haptic_input.robot_sensed_moment = -1 * get_sensed_moment(m, d, true);
}

void send_haptic_commands() {
  redis_client.setEigen(createRedisKey(COMMANDED_FORCE_KEY_SUFFIX, 0),
                        haptic_output.device_command_force);
  redis_client.setEigen(createRedisKey(COMMANDED_TORQUE_KEY_SUFFIX, 0),
                        haptic_output.device_command_moment);
}

void simulation_thread_main(SimulationLoopStats* simulation_loop_stats) {
  if (!simulation_loop_stats) {
    return;
  }

  simulation_loop_stats->sim_start_time = d->time;
  const auto wall_start_time = std::chrono::steady_clock::now();
  std::uint64_t current_reset_epoch = 0;

  while (!shutdown_requested.load()) {
    if (reset_requested.exchange(false)) {
      reset_to_home();
      ++current_reset_epoch;
      publish_snapshot_from_sim_state(current_reset_epoch);
    }

    mj_step(m, d);
    ++simulation_loop_stats->physics_step_count;
    publish_snapshot_from_sim_state(current_reset_epoch);

    const double target_sim_elapsed =
        static_cast<double>(d->time - simulation_loop_stats->sim_start_time);
    const double wall_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start_time)
                                    .count();
    const double ahead_seconds = target_sim_elapsed - wall_elapsed;
    if (ahead_seconds > 0.0) {
      std::this_thread::sleep_for(
          std::chrono::duration<double>(std::min(ahead_seconds, 0.001)));
    }
  }

  simulation_loop_stats->sim_end_time = d->time;
}

void camera_thread_main(GLFWwindow* hidden_camera_window) {
  if (!hidden_camera_window || simulation_contract.cameras.empty()) {
    return;
  }

  mjData* camera_data = nullptr;
  mjvScene camera_scene;
  mjv_defaultScene(&camera_scene);
  mjrContext camera_context;
  mjr_defaultContext(&camera_context);

  try {
    glfwMakeContextCurrent(hidden_camera_window);
    glfwSwapInterval(0);

    camera_data = mj_makeData(m);
    if (!camera_data) {
      throw std::runtime_error("Failed to allocate MuJoCo camera data.");
    }

    mjv_makeScene(m, &camera_scene, kSceneMaxGeometry);
    mjr_makeContext(m, &camera_context, mjFONTSCALE_150);
    mjr_resizeOffscreen(m->vis.global.offwidth, m->vis.global.offheight,
                        &camera_context);
    mjr_setBuffer(mjFB_OFFSCREEN, &camera_context);

    SaiCommon::RedisClient camera_redis_client("sai");
    camera_redis_client.connect();

    RenderSnapshot snapshot;
    std::uint64_t last_snapshot_seq = 0;
    std::uint64_t last_reset_epoch = std::numeric_limits<std::uint64_t>::max();
    mjvCamera capture_camera;
    mjv_defaultCamera(&capture_camera);

    while (snapshot_broker->wait_for_newer(last_snapshot_seq, snapshot,
                                           shutdown_requested)) {
      last_snapshot_seq = snapshot.seq;
      copy_snapshot_into_data(snapshot, camera_data);
      rebuild_render_data(m, camera_data);

      if (snapshot.reset_epoch != last_reset_epoch) {
        for (auto& camera : simulation_contract.cameras) {
          camera.next_publish_sim_time = snapshot.sim_time;
        }
        last_reset_epoch = snapshot.reset_epoch;
      }

      for (auto& camera : simulation_contract.cameras) {
        if (snapshot.sim_time + 1e-9 < camera.next_publish_sim_time) {
          continue;
        }

        const mjtNum publish_period = 1.0 / camera.fps;
        const mjtNum behind_time =
            std::max<mjtNum>(0.0, snapshot.sim_time - camera.next_publish_sim_time);
        const std::uint64_t missed_slots =
            publish_period > 0.0
                ? static_cast<std::uint64_t>(std::floor(
                      static_cast<double>(behind_time / publish_period)))
                : 0;
        camera.dropped_publish_slots += missed_slots;

        capture_camera.type = mjCAMERA_FIXED;
        capture_camera.fixedcamid = camera.model_camera_id;

        const mjrRect viewport = {0, 0, camera.width, camera.height};
        const auto publish_start_time = std::chrono::steady_clock::now();
        const auto render_start_time = publish_start_time;
        mjv_updateScene(m, camera_data, &opt, nullptr, &capture_camera,
                        mjCAT_ALL, &camera_scene);
        mjr_render(viewport, &camera_scene, &camera_context);
        const auto render_end_time = std::chrono::steady_clock::now();

        const auto readback_start_time = render_end_time;
        mjr_readPixels(camera.rgb_buffer.data(), nullptr, viewport,
                       &camera_context);
        const auto readback_end_time = std::chrono::steady_clock::now();

        flip_rgb_image_vertically(camera.rgb_buffer, camera.flipped_rgb_buffer,
                                  camera.width, camera.height,
                                  camera.channels);
        cv::Mat rgb_view(camera.height, camera.width, CV_8UC3,
                         camera.flipped_rgb_buffer.data());
        cv::Mat bgr_view(camera.height, camera.width, CV_8UC3,
                         camera.bgr_buffer.data());
        cv::cvtColor(rgb_view, bgr_view, cv::COLOR_RGB2BGR);

        const auto jpeg_start_time = std::chrono::steady_clock::now();
        if (!cv::imencode(".jpg", bgr_view, camera.encoded_image_buffer,
                          {cv::IMWRITE_JPEG_QUALITY, 90})) {
          throw std::runtime_error("Failed to JPEG-encode camera `" +
                                   camera.mujoco_camera_name + "`.");
        }
        const auto jpeg_end_time = std::chrono::steady_clock::now();

        const auto redis_start_time = jpeg_end_time;
        camera_redis_client.set(
            camera.redis_key,
            std::string(
                reinterpret_cast<const char*>(camera.encoded_image_buffer.data()),
                camera.encoded_image_buffer.size()));
        camera_redis_client.set(camera.metadata_redis_key,
                                make_camera_metadata_json(camera, snapshot));
        const auto publish_end_time = std::chrono::steady_clock::now();

        camera.total_render_seconds +=
            std::chrono::duration<double>(render_end_time - render_start_time)
                .count();
        camera.total_readback_seconds +=
            std::chrono::duration<double>(readback_end_time - readback_start_time)
                .count();
        camera.total_jpeg_encode_seconds +=
            std::chrono::duration<double>(jpeg_end_time - jpeg_start_time)
                .count();
        camera.total_redis_publish_seconds +=
            std::chrono::duration<double>(publish_end_time - redis_start_time)
                .count();
        camera.total_publish_seconds +=
            std::chrono::duration<double>(publish_end_time - publish_start_time)
                .count();
        ++camera.publish_count;
        camera.next_publish_sim_time +=
            static_cast<mjtNum>(missed_slots + 1) * publish_period;
      }
    }
  } catch (const std::exception& exception) {
    std::cerr << "Camera publisher thread failed: " << exception.what()
              << "\n";
    shutdown_requested = true;
    if (snapshot_broker) {
      snapshot_broker->notify_all();
    }
  }

  mjv_freeScene(&camera_scene);
  mjr_freeContext(&camera_context);
  if (camera_data) {
    mj_deleteData(camera_data);
  }
  glfwMakeContextCurrent(nullptr);
}

// -------- Redis Code ---------------------------------

int main(int argc, char** argv) {
  const auto startup_options = parse_startup_options(argc, argv);
  if (!startup_options) {
    return 1;
  }

  is_data_collection = startup_options->is_data_collection;
  const fs::path contract_path = startup_options->contract_path;
  try {
    simulation_contract = load_simulation_contract(contract_path);
  } catch (const std::exception& exception) {
    std::cerr << "Failed to load simulation contract from "
              << contract_path.string() << ": " << exception.what() << "\n";
    return 1;
  }

  const string mujoco_file = simulation_contract.xml_path.string();
  const string robot_file = simulation_contract.urdf_path.string();

  std::cout << "Contract: " << contract_path.string() << "\n";
  std::cout << "Mode: "
            << (is_data_collection ? "data_collection" : "inference")
            << "\n";
  std::cout << "MuJoCo xml: " << mujoco_file << "\n";
  std::cout << "Robot urdf: " << robot_file << "\n";

  load_mujoco_plugins();

  char error[1000] = "Could not load MuJoCo model";
  m = mj_loadXML(mujoco_file.c_str(), nullptr, error, sizeof(error));
  if (!m) {
    std::cerr << "Failed to load model: " << error << "\n";
    return 1;
  }

  d = mj_makeData(m);
  if (!d) {
    std::cerr << "Failed to create MuJoCo data.\n";
    mj_deleteModel(m);
    return 1;
  }

  if (!initialize_force_sensor_handles(m)) {
    mj_deleteData(d);
    mj_deleteModel(m);
    return 1;
  }

  try {
    preflight_camera_publishers();
  } catch (const std::exception& exception) {
    std::cerr << "Camera preflight failed: " << exception.what() << "\n";
    mj_deleteData(d);
    mj_deleteModel(m);
    return 1;
  }

  // Disabling the joint limits so the controller can come up with torques
  for (int i = 0; i < m->njnt; ++i) {
        m->jnt_range[2 * i] = -1e10;     // Lower limit
        m->jnt_range[2 * i + 1] = 1e10; // Upper limit
    }

  std::cout << "Joint limits disabled." << std::endl;

  // loading up the redis client 
  redis_client.connect();
  init_redis();

  //resetting the robot to the home position
  reset_to_home();

  //Creating all the opensai related objects and controllers --------------------
  robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
  std::cout << "Robot DOF: " << robot->dof() << "\n";
  std::cout << "MJ DOF: " << m->nq << "\n"; 
  control_point = Vector3d(0.0, 0.0, 0.0);
  control_frame = Affine3d::Identity();
  control_frame.translation() = control_point;
  motion_force_task = std::make_shared<SaiPrimitives::MotionForceTask>(robot, control_link, control_frame);
  motion_force_task->setPosControlGains(400.0, 40.0);
  
  motion_force_task->disableInternalOtg();
  joint_task = std::make_shared<SaiPrimitives::JointTask>(robot);
  update_robot_state(m, d);
  motion_force_task->reInitializeTask();
  joint_task->reInitializeTask();
  // ----------------------------------------------------------------------------

  if (is_data_collection) {
    try {
      SaiPrimitives::HapticDeviceController::DeviceLimits device_limits(
          redis_client.getEigen(createRedisKey(MAX_STIFFNESS_KEY_SUFFIX, 0)),
          redis_client.getEigen(createRedisKey(MAX_DAMPING_KEY_SUFFIX, 0)),
          redis_client.getEigen(createRedisKey(MAX_FORCE_KEY_SUFFIX, 0)));

      haptic_controller =
          std::make_shared<SaiPrimitives::HapticDeviceController>(
              device_limits, robot->transformInWorld(control_link));

      haptic_controller->setScalingFactors(3.5);
      haptic_controller->setHapticControlType(
          SaiPrimitives::HapticControlType::MOTION_MOTION);
      directions_of_proxy_feedback = Vector3i::Zero();
      prev_sensed_force = Vector3d::Zero();

      haptic_controller->setDeviceControlGains(
        0.1 * device_limits.max_linear_stiffness,
        0.3 * device_limits.max_linear_damping,
        0.1 * device_limits.max_angular_stiffness,
        0.3 * device_limits.max_angular_damping);

      haptic_controller->setReductionFactorForce(0.2);
      haptic_controller->setReductionFactorMoment(0.2);
    } catch (const std::exception& exception) {
      std::cerr << exception.what() << "\n";
      mj_deleteData(d);
      mj_deleteModel(m);
      return 1;
    }
  }

  //initialize the haptic controller ----- ----------------
  

  mjcb_control = controller_callback;

  snapshot_broker = std::make_unique<SnapshotBroker>(m);
  viewer_data = mj_makeData(m);
  if (!viewer_data) {
    std::cerr << "Failed to create MuJoCo viewer data.\n";
    mjcb_control = nullptr;
    mj_deleteData(d);
    mj_deleteModel(m);
    return 1;
  }
  mj_resetData(m, viewer_data);
  publish_snapshot_from_sim_state(0);

  if (!glfwInit()) {
    std::cerr << "Could not initialize GLFW.\n";
    mj_deleteData(viewer_data);
    mj_deleteData(d);
    mj_deleteModel(m);
    return 1;
  }

  GLFWwindow* window =
      glfwCreateWindow(kWindowWidth, kWindowHeight, "FR3 Viewer",
                       nullptr, nullptr);
  if (!window) {
    std::cerr << "Could not create GLFW window.\n";
    mj_deleteData(viewer_data);
    mj_deleteData(d);
    mj_deleteModel(m);
    glfwTerminate();
    return 1;
  }

  GLFWwindow* hidden_camera_window = nullptr;
  if (!simulation_contract.cameras.empty()) {
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    hidden_camera_window =
        glfwCreateWindow(1, 1, "ForceWM Camera Publisher", nullptr, window);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    if (!hidden_camera_window) {
      std::cerr << "Could not create hidden camera publishing window.\n";
      glfwDestroyWindow(window);
      mj_deleteData(viewer_data);
      mj_deleteData(d);
      mj_deleteModel(m);
      glfwTerminate();
      return 1;
    }
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);

  initialize_camera();
  mjv_makeScene(m, &scn, kSceneMaxGeometry);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  RenderSnapshot viewer_snapshot;
  std::uint64_t last_viewer_snapshot_seq = 0;
  if (snapshot_broker->copy_latest(viewer_snapshot)) {
    copy_snapshot_into_data(viewer_snapshot, viewer_data);
    rebuild_render_data(m, viewer_data);
    last_viewer_snapshot_seq = viewer_snapshot.seq;
  }

  glfwSetKeyCallback(window, keyboard);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetScrollCallback(window, scroll);

  std::cout << "Viewer controls: left drag = rotate, right drag = pan, "
               "scroll = zoom, Backspace = reset, Esc = quit.\n";

  const auto wall_start_time = std::chrono::steady_clock::now();
  SimulationLoopStats simulation_loop_stats;
  std::uint64_t rendered_frame_count = 0;

  std::thread simulation_thread(simulation_thread_main, &simulation_loop_stats);
  std::thread camera_thread;
  if (hidden_camera_window) {
    camera_thread = std::thread(camera_thread_main, hidden_camera_window);
  }

  while (!glfwWindowShouldClose(window) && !shutdown_requested.load()) {
    if (snapshot_broker->copy_latest(viewer_snapshot) &&
        viewer_snapshot.seq != last_viewer_snapshot_seq) {
      copy_snapshot_into_data(viewer_snapshot, viewer_data);
      rebuild_render_data(m, viewer_data);
      last_viewer_snapshot_seq = viewer_snapshot.seq;
    }

    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    mjv_updateScene(m, viewer_data, &opt, nullptr, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    glfwSwapBuffers(window);
    glfwPollEvents();
    ++rendered_frame_count;
  }

  shutdown_requested = true;
  if (snapshot_broker) {
    snapshot_broker->notify_all();
  }

  if (simulation_thread.joinable()) {
    simulation_thread.join();
  }
  if (camera_thread.joinable()) {
    camera_thread.join();
  }

  const auto wall_end_time = std::chrono::steady_clock::now();
  SimulationPerformanceStats performance_stats;
  performance_stats.physics_step_count = simulation_loop_stats.physics_step_count;
  performance_stats.rendered_frame_count = rendered_frame_count;
  performance_stats.simulated_seconds =
      simulation_loop_stats.sim_end_time - simulation_loop_stats.sim_start_time;
  performance_stats.wall_seconds =
      std::chrono::duration<double>(wall_end_time - wall_start_time).count();

  mjv_freeScene(&scn);
  mjr_freeContext(&con);
  if (hidden_camera_window) {
    glfwDestroyWindow(hidden_camera_window);
  }
  glfwDestroyWindow(window);
  mjcb_control = nullptr;
  mj_deleteData(viewer_data);
  mj_deleteData(d);
  mj_deleteModel(m);

#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif

  print_simulation_summary(performance_stats);
  print_camera_publish_summary();

  return 0;
}
