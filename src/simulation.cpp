#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
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
mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjvScene camera_scn;
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

struct CameraStreamConfig {
  std::string redis_key;
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

SimulationContractConfig simulation_contract;

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

    if (camera.fps > (1.0 / kRenderTimestep)) {
      std::cout << "Camera '" << camera.mujoco_camera_name << "' requests "
                << camera.fps
                << " Hz, but render-loop publishing is currently capped by the "
                   "viewer rate.\n";
    }

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
    reset_to_home();
  } else if (act == GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
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

void update_camera_redis_keys() {
  if (simulation_contract.cameras.empty()) {
    return;
  }

  bool rendered_any_camera = false;
  mjvCamera capture_camera;
  mjv_defaultCamera(&capture_camera);

  for (auto& camera : simulation_contract.cameras) {
    if (d->time + 1e-9 < camera.next_publish_sim_time) {
      continue;
    }

    if (!rendered_any_camera) {
      mjr_setBuffer(mjFB_OFFSCREEN, &con);
      rendered_any_camera = true;
    }

    capture_camera.type = mjCAMERA_FIXED;
    capture_camera.fixedcamid = camera.model_camera_id;

    const mjrRect viewport = {0, 0, camera.width, camera.height};
    mjv_updateScene(m, d, &opt, nullptr, &capture_camera, mjCAT_ALL,
                    &camera_scn);
    mjr_render(viewport, &camera_scn, &con);
    mjr_readPixels(camera.rgb_buffer.data(), nullptr, viewport, &con);

    flip_rgb_image_vertically(camera.rgb_buffer, camera.flipped_rgb_buffer,
                              camera.width, camera.height, camera.channels);
    cv::Mat rgb_view(camera.height, camera.width, CV_8UC3,
                     camera.flipped_rgb_buffer.data());
    cv::Mat bgr_view(camera.height, camera.width, CV_8UC3,
                     camera.bgr_buffer.data());
    cv::cvtColor(rgb_view, bgr_view, cv::COLOR_RGB2BGR);

    if (!cv::imencode(".jpg", bgr_view, camera.encoded_image_buffer,
                      {cv::IMWRITE_JPEG_QUALITY, 90})) {
      throw std::runtime_error("Failed to JPEG-encode camera `" +
                               camera.mujoco_camera_name + "`.");
    }

    redis_client.set(camera.redis_key,
                     std::string(
                         reinterpret_cast<const char*>(
                             camera.encoded_image_buffer.data()),
                         camera.encoded_image_buffer.size()));

    const mjtNum publish_period = 1.0 / camera.fps;
    do {
      camera.next_publish_sim_time += publish_period;
    } while (camera.next_publish_sim_time <= d->time);
  }

  if (rendered_any_camera) {
    mjr_setBuffer(mjFB_WINDOW, &con);
  }
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

  if (!glfwInit()) {
    std::cerr << "Could not initialize GLFW.\n";
    mj_deleteData(d);
    mj_deleteModel(m);
    return 1;
  }

  GLFWwindow* window =
      glfwCreateWindow(kWindowWidth, kWindowHeight, "FR3 Viewer",
                       nullptr, nullptr);
  if (!window) {
    std::cerr << "Could not create GLFW window.\n";
    mj_deleteData(d);
    mj_deleteModel(m);
    glfwTerminate();
    return 1;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjv_defaultScene(&camera_scn);
  mjr_defaultContext(&con);

  initialize_camera();
  mjv_makeScene(m, &scn, kSceneMaxGeometry);
  if (!simulation_contract.cameras.empty()) {
    mjv_makeScene(m, &camera_scn, kSceneMaxGeometry);
  }
  mjr_makeContext(m, &con, mjFONTSCALE_150);
  if (!simulation_contract.cameras.empty()) {
    mjr_resizeOffscreen(m->vis.global.offwidth, m->vis.global.offheight, &con);
  }

  glfwSetKeyCallback(window, keyboard);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetScrollCallback(window, scroll);

  std::cout << "Viewer controls: left drag = rotate, right drag = pan, "
               "scroll = zoom, Backspace = reset, Esc = quit.\n";

  const auto wall_start_time = std::chrono::steady_clock::now();
  const mjtNum sim_start_time = d->time;
  std::uint64_t physics_step_count = 0;
  std::uint64_t rendered_frame_count = 0;

  while (!glfwWindowShouldClose(window)) {
    const mjtNum simstart = d->time;

    while (d->time - simstart < kRenderTimestep) {
      mj_step(m, d);
      ++physics_step_count;
    }

    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // update_camera_redis_keys();
    mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    glfwSwapBuffers(window);
    glfwPollEvents();
    ++rendered_frame_count;
  }

  const auto wall_end_time = std::chrono::steady_clock::now();
  SimulationPerformanceStats performance_stats;
  performance_stats.physics_step_count = physics_step_count;
  performance_stats.rendered_frame_count = rendered_frame_count;
  performance_stats.simulated_seconds = d->time - sim_start_time;
  performance_stats.wall_seconds =
      std::chrono::duration<double>(wall_end_time - wall_start_time).count();

  mjv_freeScene(&scn);
  mjv_freeScene(&camera_scn);
  mjr_freeContext(&con);
  mjcb_control = nullptr;
  mj_deleteData(d);
  mj_deleteModel(m);

#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif

  print_simulation_summary(performance_stats);

  return 0;
}
