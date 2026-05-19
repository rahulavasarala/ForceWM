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
#include <unordered_map>
#include <vector>

#include "SaiModel.h"
#include "SaiPrimitives.h"
#include "control_utils.h"
#include "particle_filter/ForceSpaceParticleFilter.h"

#include "redis/RedisClient.h"
#include "redis_keys.h"
#include "utils.h"

SaiCommon::RedisClient redis_client = SaiCommon::RedisClient("sai");;

using std::string;
namespace fs = std::filesystem;

namespace {

constexpr int kWindowWidth = 1200;
constexpr int kWindowHeight = 900;
constexpr int kSceneMaxGeometry = 2000;
constexpr mjtNum kRenderTimestep = 1.0 / 60.0;
constexpr int kRobotDof = 7;
constexpr double kSensedWrenchLowPassAlpha = 0.2;
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

Vector3d START_POS = Vector3d(0.4, 0.0, 0.39);
Matrix3d START_ORIENTATION = (Matrix3d() << 
    1,  0,  0,
    0, -1,  0,
    0,  0, -1).finished();

std::shared_ptr<SaiModel::SaiModel> robot;
std::shared_ptr<SaiPrimitives::MotionForceTask> motion_force_task;
std::shared_ptr<SaiPrimitives::JointTask> joint_task;

// Init the force space particle filter ---- 
std::shared_ptr<ForceWM::ForceSpaceParticleFilter> force_space_particle_filter;
std::queue<int> force_dimension_queue;
Matrix3d sigma_force = Matrix3d::Zero();
Matrix3d sigma_motion = Matrix3d::Identity();
PFilterOutput pfilter_output;

std::shared_ptr<SaiPrimitives::HapticDeviceController> haptic_controller;
SaiPrimitives::HapticControllerInput haptic_input;
SaiPrimitives::HapticControllerOutput haptic_output;

Vector3d prev_sensed_force;
Vector3d filtered_sensed_force_sensor_frame = Vector3d::Zero();
Vector3d filtered_sensed_moment_sensor_frame = Vector3d::Zero();
bool sensed_wrench_filter_initialized = false;
Vector3i directions_of_proxy_feedback;

Vector3d control_point;
Affine3d control_frame;

std::string control_link = "fr3_link8";
int ee_force_sensor_id = -1;
int ee_torque_sensor_id = -1;
int ee_sensor_site_id = -1;
std::atomic<bool> shutdown_requested = false;
std::atomic<bool> reset_requested = false;

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

  bool wait_for_first_snapshot(RenderSnapshot& out_snapshot,
                               const std::atomic<bool>& stop_flag) const {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return stop_flag.load() || has_snapshot; });

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
    if (camera.aligned_depth) {
      std::cout << "    Aligned depth -> `" << camera.aligned_depth->redis_key
                << "`\n";
    }

    if (camera.publish_count == 0) {
      std::cout << "    No images published.\n";
      continue;
    }

    const double publish_count = static_cast<double>(camera.publish_count);
    const double avg_scene_update_ms =
        1000.0 * camera.total_scene_update_seconds / publish_count;
    const double avg_render_ms =
        1000.0 * camera.total_render_seconds / publish_count;
    const double avg_render_draw_ms =
        1000.0 * camera.total_render_draw_seconds / publish_count;
    const double avg_readback_ms =
        1000.0 * camera.total_readback_seconds / publish_count;
    const double avg_flip_ms =
        1000.0 * camera.total_flip_seconds / publish_count;
    const double avg_color_convert_ms =
        1000.0 * camera.total_color_convert_seconds / publish_count;
    const double avg_frame_encode_ms =
        1000.0 * camera.total_frame_encode_seconds / publish_count;
    const double avg_redis_ms =
        1000.0 * camera.total_redis_publish_seconds / publish_count;
    const double avg_total_ms =
        1000.0 * camera.total_publish_seconds / publish_count;

    std::cout << "    Published frames: " << camera.publish_count << "\n";
    std::cout << "    Dropped publish slots: " << camera.dropped_publish_slots
              << "\n";
    std::cout << "    Avg scene update time: " << avg_scene_update_ms
              << " ms\n";
    std::cout << "    Avg render time: " << avg_render_ms << " ms\n";
    std::cout << "    Avg draw time: " << avg_render_draw_ms << " ms\n";
    std::cout << "    Avg readback time: " << avg_readback_ms << " ms\n";
    std::cout << "    Avg image flip time: " << avg_flip_ms << " ms\n";
    std::cout << "    Avg color convert time: " << avg_color_convert_ms
              << " ms\n";
    std::cout << "    Avg frame encode time: " << avg_frame_encode_ms
              << " ms\n";
    std::cout << "    Avg Redis publish time: " << avg_redis_ms << " ms\n";
    std::cout << "    Avg total publish time: " << avg_total_ms << " ms\n";
  }

  std::cout << std::defaultfloat;
}

const char* visual_stream_type_name(const VisualStreamType type) {
  switch (type) {
    case VisualStreamType::kRgb:
      return "rgb";
    case VisualStreamType::kDepth:
      return "depth";
  }

  return "unknown";
}

double wall_time_now_seconds() {
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

template <typename T>
void flip_image_vertically(const std::vector<T>& source,
                           std::vector<T>& destination,
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

void convert_mujoco_depth_to_millimeters(
    const std::vector<float>& source_depth,
    std::vector<std::uint16_t>& destination_depth_mm,
    const float near_clip_m,
    const float far_clip_m) {
  destination_depth_mm.resize(source_depth.size());
  const double clip_ratio = 1.0 - static_cast<double>(near_clip_m / far_clip_m);

  for (size_t depth_index = 0; depth_index < source_depth.size();
       ++depth_index) {
    const float raw_depth = source_depth[depth_index];
    std::uint16_t depth_mm = 0;

    if (std::isfinite(raw_depth) && raw_depth >= 0.0f && raw_depth < 1.0f) {
      const double denominator =
          1.0 - static_cast<double>(raw_depth) * clip_ratio;
      if (denominator > 1e-12) {
        const double depth_m =
            static_cast<double>(near_clip_m) / denominator;
        const double depth_mm_value = depth_m * 1000.0;
        if (std::isfinite(depth_mm_value) && depth_mm_value > 0.0 &&
            depth_mm_value <=
                static_cast<double>(std::numeric_limits<std::uint16_t>::max())) {
          depth_mm = static_cast<std::uint16_t>(std::llround(depth_mm_value));
        }
      }
    }

    destination_depth_mm[depth_index] = depth_mm;
  }
}

std::string make_rgb_camera_metadata_json(const CameraStreamConfig& camera,
                                          const RenderSnapshot& snapshot) {
  std::ostringstream metadata_stream;
  metadata_stream << std::fixed << std::setprecision(17);
  metadata_stream << '{';
  metadata_stream << '"' << "seq" << '"' << ':' << snapshot.seq;
  metadata_stream << ',' << '"' << "reset_epoch" << '"' << ':'
                  << snapshot.reset_epoch;
  metadata_stream << ',' << '"' << "contract_key" << '"' << ':' << '"'
                  << camera.visual_name << '"';
  metadata_stream << ',' << '"' << "type" << '"' << ':' << '"'
                  << visual_stream_type_name(camera.type) << '"';
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
                  << camera.encoding << '"';
  if (camera.encoding == "jpeg") {
    metadata_stream << ',' << '"' << "jpeg_quality" << '"' << ':' << 90;
  }
  metadata_stream << ',' << '"' << "dropped_slots_total" << '"' << ':'
                  << camera.dropped_publish_slots;
  metadata_stream << '}';
  return metadata_stream.str();
}

std::string make_depth_camera_metadata_json(const CameraStreamConfig& camera,
                                            const DepthStreamConfig& depth_stream,
                                            const RenderSnapshot& snapshot) {
  std::ostringstream metadata_stream;
  metadata_stream << std::fixed << std::setprecision(17);
  metadata_stream << '{';
  metadata_stream << '"' << "seq" << '"' << ':' << snapshot.seq;
  metadata_stream << ',' << '"' << "reset_epoch" << '"' << ':'
                  << snapshot.reset_epoch;
  metadata_stream << ',' << '"' << "contract_key" << '"' << ':' << '"'
                  << depth_stream.visual_name << '"';
  metadata_stream << ',' << '"' << "type" << '"' << ':' << '"'
                  << visual_stream_type_name(VisualStreamType::kDepth) << '"';
  metadata_stream << ',' << '"' << "camera_name" << '"' << ':' << '"'
                  << camera.mujoco_camera_name << '"';
  metadata_stream << ',' << '"' << "align_to" << '"' << ':' << '"'
                  << depth_stream.align_to_visual_name << '"';
  metadata_stream << ',' << '"' << "sim_time_s" << '"' << ':'
                  << snapshot.sim_time;
  metadata_stream << ',' << '"' << "publish_wall_time_s" << '"' << ':'
                  << snapshot.publish_wall_time_s;
  metadata_stream << ',' << '"' << "width" << '"' << ':' << depth_stream.width;
  metadata_stream << ',' << '"' << "height" << '"' << ':'
                  << depth_stream.height;
  metadata_stream << ',' << '"' << "channels" << '"' << ':'
                  << depth_stream.channels;
  metadata_stream << ',' << '"' << "encoding" << '"' << ':' << '"'
                  << depth_stream.encoding << '"';
  metadata_stream << ',' << '"' << "unit" << '"' << ':' << '"'
                  << depth_stream.unit << '"';
  metadata_stream << ',' << '"' << "dropped_slots_total" << '"' << ':'
                  << camera.dropped_publish_slots;
  metadata_stream << '}';
  return metadata_stream.str();
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
              << camera.width << "x" << camera.height << ").";
    if (camera.aligned_depth) {
      std::cout << " Aligned depth -> Redis key `"
                << camera.aligned_depth->redis_key << "`";
    }
    std::cout << "\n";
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

  prev_sensed_force = Vector3d::Zero();
  filtered_sensed_force_sensor_frame = Vector3d::Zero();
  filtered_sensed_moment_sensor_frame = Vector3d::Zero();
  sensed_wrench_filter_initialized = false;
  directions_of_proxy_feedback = Vector3i::Zero();
  sigma_force = Matrix3d::Zero();
  sigma_motion = Matrix3d::Identity();
  pfilter_output = PFilterOutput{};
  if (force_space_particle_filter) {
    force_space_particle_filter->reset();
  }

  const std::size_t queue_size = force_dimension_queue.size();
  while (!force_dimension_queue.empty()) {
    force_dimension_queue.pop();
  }
  for (std::size_t i = 0; i < queue_size; ++i) {
    force_dimension_queue.push(0);
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

void inference_time_callback(const mjModel* m, mjData* d) {

  //When the robot is in inference mode, we have things that are ---- 
  // only relevant to inference and not data collection.

  static int particle_filter_count = 0;

  if (particle_filter_count % 67 == 0) {  // Update the particle filter every 67 control steps (approximately every 1 second at 15ms control timestep)
    update_particle_filter(
        motion_force_task, force_space_particle_filter, pfilter_output,
        force_dimension_queue, sigma_force, sigma_motion, m, d, ee_force_sensor_id,
        ee_sensor_site_id, filtered_sensed_force_sensor_frame,
        sensed_wrench_filter_initialized);
    particle_filter_count = 0;
  }

  particle_filter_count++;

  update_robot_state(m, d);
  update_filtered_sensed_wrench(
      m, d, ee_force_sensor_id, ee_torque_sensor_id, ee_sensor_site_id,
      kSensedWrenchLowPassAlpha, filtered_sensed_force_sensor_frame,
      filtered_sensed_moment_sensor_frame, sensed_wrench_filter_initialized);
  update_redis(redis_client, motion_force_task, force_space_particle_filter,
               pfilter_output, m,
               d, kRobotDof, ee_force_sensor_id, ee_torque_sensor_id,
               ee_sensor_site_id, filtered_sensed_force_sensor_frame,
               filtered_sensed_moment_sensor_frame,
               sensed_wrench_filter_initialized);
  query_redis_for_desired_state(redis_client, motion_force_task,
                                pfilter_output.force_space_dimension,
                                pfilter_output.force_or_motion_axis);

  motion_force_task->updateTaskModel(MatrixXd::Identity(robot->dof(), robot->dof()));
  joint_task->updateTaskModel(motion_force_task->getTaskAndPreviousNullspace());
  const VectorXd control_torques =
      motion_force_task->computeTorques() + joint_task->computeTorques() +
      robot->jointGravityVector();

  for (int i = 0; i < std::min<int>(control_torques.size(), m->nu); ++i) {
    d->ctrl[i] = control_torques(i);
  }

}

void data_collection_time_callback(const mjModel* m, mjData* d) {

  //If inference mode is false ---- then we are in data collection mo

  update_robot_state(m, d);
  update_filtered_sensed_wrench(
      m, d, ee_force_sensor_id, ee_torque_sensor_id, ee_sensor_site_id,
      kSensedWrenchLowPassAlpha, filtered_sensed_force_sensor_frame,
      filtered_sensed_moment_sensor_frame, sensed_wrench_filter_initialized);
  update_redis(redis_client, motion_force_task, force_space_particle_filter,
               pfilter_output, m,
               d, kRobotDof, ee_force_sensor_id, ee_torque_sensor_id,
               ee_sensor_site_id, filtered_sensed_force_sensor_frame,
               filtered_sensed_moment_sensor_frame,
               sensed_wrench_filter_initialized);
  update_haptic_information(robot, control_link, redis_client, m, d,
                            ee_force_sensor_id, ee_torque_sensor_id,
                            ee_sensor_site_id,
                            filtered_sensed_force_sensor_frame,
                            filtered_sensed_moment_sensor_frame,
                            sensed_wrench_filter_initialized, haptic_input);
  haptic_output = haptic_controller->computeHapticControl(haptic_input);

  send_haptic_commands(redis_client, haptic_output);

  motion_force_task->updateSensedForceAndMoment(
      -1 * get_filtered_sensed_force(
               m, d, ee_force_sensor_id, ee_sensor_site_id,
               filtered_sensed_force_sensor_frame,
               sensed_wrench_filter_initialized),
      -1 * get_filtered_sensed_moment(
               m, d, ee_torque_sensor_id, ee_sensor_site_id,
               filtered_sensed_moment_sensor_frame,
               sensed_wrench_filter_initialized));

  motion_force_task->setGoalPosition(haptic_output.robot_goal_position);
  motion_force_task->setGoalOrientation(
    haptic_output.robot_goal_orientation);

  motion_force_task->updateTaskModel(MatrixXd::Identity(robot->dof(), robot->dof()));
  joint_task->updateTaskModel(motion_force_task->getTaskAndPreviousNullspace());

  const VectorXd control_torques =
      motion_force_task->computeTorques() + joint_task->computeTorques() +
      robot->jointGravityVector();

  Vector3d sensed_force_world_frame =
      -1 * get_filtered_sensed_force(
               m, d, ee_force_sensor_id, ee_sensor_site_id,
               filtered_sensed_force_sensor_frame,
               sensed_wrench_filter_initialized, true);

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

  update_reset_request_from_redis(redis_client, reset_requested);

  if (is_data_collection) {
    data_collection_time_callback(m, d);
  } else {
    inference_time_callback(m, d);
  }
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
    std::uint64_t last_reset_epoch = std::numeric_limits<std::uint64_t>::max();
    mjvCamera capture_camera;
    mjv_defaultCamera(&capture_camera);
    std::vector<double> next_camera_publish_wall_times(
        simulation_contract.cameras.size(), 0.0);
    const float depth_near_clip_m =
        static_cast<float>(m->vis.map.znear * m->stat.extent);
    const float depth_far_clip_m =
        static_cast<float>(m->vis.map.zfar * m->stat.extent);
    if (!(depth_near_clip_m > 0.0f && depth_far_clip_m > depth_near_clip_m)) {
      throw std::runtime_error(
          "Invalid MuJoCo near/far clipping settings for depth publishing.");
    }

    if (!snapshot_broker->wait_for_first_snapshot(snapshot, shutdown_requested)) {
      return;
    }

    copy_snapshot_into_data(snapshot, camera_data);
    rebuild_render_data(m, camera_data);

    while (!shutdown_requested.load()) {
      const double now = wall_time_now_seconds();
      double next_deadline = std::numeric_limits<double>::infinity();
      bool should_publish_any_camera = false;

      for (size_t camera_index = 0; camera_index < simulation_contract.cameras.size();
           ++camera_index) {
        const auto& camera = simulation_contract.cameras[camera_index];
        if (next_camera_publish_wall_times[camera_index] <= 0.0) {
          next_camera_publish_wall_times[camera_index] = now;
        }

        next_deadline =
            std::min(next_deadline, next_camera_publish_wall_times[camera_index]);
        if (next_camera_publish_wall_times[camera_index] <= now) {
          should_publish_any_camera = true;
        }
      }

      if (!should_publish_any_camera) {
        const double sleep_seconds = next_deadline - now;
        if (sleep_seconds > 0.0) {
          std::this_thread::sleep_for(std::chrono::duration<double>(
              std::min(sleep_seconds, 0.001)));
        }
        continue;
      }

      if (!snapshot_broker->copy_latest(snapshot)) {
        continue;
      }

      copy_snapshot_into_data(snapshot, camera_data);
      rebuild_render_data(m, camera_data);

      if (snapshot.reset_epoch != last_reset_epoch) {
        const double reset_wall_time = wall_time_now_seconds();
        for (size_t camera_index = 0; camera_index < simulation_contract.cameras.size();
             ++camera_index) {
          auto& camera = simulation_contract.cameras[camera_index];
          camera.next_publish_sim_time = snapshot.sim_time;
          next_camera_publish_wall_times[camera_index] = reset_wall_time;
        }
        last_reset_epoch = snapshot.reset_epoch;
      }

      const double publish_loop_now = wall_time_now_seconds();
      for (size_t camera_index = 0; camera_index < simulation_contract.cameras.size();
           ++camera_index) {
        auto& camera = simulation_contract.cameras[camera_index];
        if (next_camera_publish_wall_times[camera_index] > publish_loop_now) {
          continue;
        }

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
        const auto scene_update_start_time = publish_start_time;
        mjv_updateScene(m, camera_data, &opt, nullptr, &capture_camera,
                        mjCAT_ALL, &camera_scene);
        const auto scene_update_end_time = std::chrono::steady_clock::now();

        const auto render_start_time = scene_update_end_time;
        mjr_render(viewport, &camera_scene, &camera_context);
        const auto render_end_time = std::chrono::steady_clock::now();

        const auto readback_start_time = render_end_time;
        float* depth_readback_buffer = nullptr;
        if (camera.aligned_depth) {
          depth_readback_buffer = camera.aligned_depth->raw_depth_buffer.data();
        }
        mjr_readPixels(camera.rgb_buffer.data(), depth_readback_buffer,
                       viewport, &camera_context);
        const auto readback_end_time = std::chrono::steady_clock::now();

        const auto flip_start_time = readback_end_time;
        flip_image_vertically(camera.rgb_buffer, camera.flipped_rgb_buffer,
                              camera.width, camera.height, camera.channels);
        if (camera.aligned_depth) {
          auto& depth_stream = *camera.aligned_depth;
          flip_image_vertically(depth_stream.raw_depth_buffer,
                                depth_stream.flipped_depth_buffer,
                                depth_stream.width, depth_stream.height,
                                depth_stream.channels);
        }
        const auto flip_end_time = std::chrono::steady_clock::now();

        cv::Mat rgb_view(camera.height, camera.width, CV_8UC3,
                         camera.flipped_rgb_buffer.data());
        cv::Mat bgr_view(camera.height, camera.width, CV_8UC3,
                         camera.bgr_buffer.data());
        const auto color_convert_start_time = flip_end_time;
        cv::cvtColor(rgb_view, bgr_view, cv::COLOR_RGB2BGR);
        const auto color_convert_end_time = std::chrono::steady_clock::now();

        const auto jpeg_start_time = color_convert_end_time;
        if (!cv::imencode(".jpg", bgr_view, camera.encoded_image_buffer,
                          {cv::IMWRITE_JPEG_QUALITY, 90})) {
          throw std::runtime_error("Failed to JPEG-encode camera `" +
                                   camera.mujoco_camera_name + "`.");
        }
        if (camera.aligned_depth) {
          auto& depth_stream = *camera.aligned_depth;
          convert_mujoco_depth_to_millimeters(
              depth_stream.flipped_depth_buffer, depth_stream.depth_mm_buffer,
              depth_near_clip_m, depth_far_clip_m);
          cv::Mat depth_view(depth_stream.height, depth_stream.width, CV_16UC1,
                             depth_stream.depth_mm_buffer.data());
          if (!cv::imencode(".png", depth_view,
                            depth_stream.encoded_depth_buffer)) {
            throw std::runtime_error("Failed to PNG16-encode depth stream `" +
                                     depth_stream.visual_name + "`.");
          }
        }
        const auto encode_end_time = std::chrono::steady_clock::now();

        const auto redis_start_time = encode_end_time;
        camera_redis_client.set(
            camera.redis_key,
            std::string(
                reinterpret_cast<const char*>(camera.encoded_image_buffer.data()),
                camera.encoded_image_buffer.size()));
        camera_redis_client.set(camera.metadata_redis_key,
                                make_rgb_camera_metadata_json(camera, snapshot));
        if (camera.aligned_depth) {
          const auto& depth_stream = *camera.aligned_depth;
          camera_redis_client.set(
              depth_stream.redis_key,
              std::string(reinterpret_cast<const char*>(
                              depth_stream.encoded_depth_buffer.data()),
                          depth_stream.encoded_depth_buffer.size()));
          camera_redis_client.set(
              depth_stream.metadata_redis_key,
              make_depth_camera_metadata_json(camera, depth_stream, snapshot));
        }
        const auto publish_end_time = std::chrono::steady_clock::now();

        camera.total_scene_update_seconds +=
            std::chrono::duration<double>(scene_update_end_time -
                                          scene_update_start_time)
                .count();
        camera.total_render_seconds +=
            std::chrono::duration<double>(render_end_time -
                                          scene_update_start_time)
                .count();
        camera.total_render_draw_seconds +=
            std::chrono::duration<double>(render_end_time - render_start_time)
                .count();
        camera.total_readback_seconds +=
            std::chrono::duration<double>(readback_end_time - readback_start_time)
                .count();
        camera.total_flip_seconds +=
            std::chrono::duration<double>(flip_end_time - flip_start_time)
                .count();
        camera.total_color_convert_seconds +=
            std::chrono::duration<double>(color_convert_end_time -
                                          color_convert_start_time)
                .count();
        camera.total_frame_encode_seconds +=
            std::chrono::duration<double>(encode_end_time - jpeg_start_time)
                .count();
        camera.total_redis_publish_seconds +=
            std::chrono::duration<double>(publish_end_time - redis_start_time)
                .count();
        camera.total_publish_seconds +=
            std::chrono::duration<double>(publish_end_time - publish_start_time)
                .count();
        ++camera.publish_count;
        next_camera_publish_wall_times[camera_index] += publish_period;
        camera.next_publish_sim_time +=
            static_cast<mjtNum>(missed_slots + 1) * publish_period;

        const double publish_end_wall_time = wall_time_now_seconds();
        if (next_camera_publish_wall_times[camera_index] < publish_end_wall_time) {
          const double wall_behind_time =
              publish_end_wall_time - next_camera_publish_wall_times[camera_index];
          const std::uint64_t extra_missed_wall_slots =
              publish_period > 0.0
                  ? static_cast<std::uint64_t>(std::floor(
                        wall_behind_time / publish_period))
                  : 0;
          next_camera_publish_wall_times[camera_index] +=
              static_cast<double>(extra_missed_wall_slots) * publish_period;
        }
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

  if (!initialize_force_sensor_handles(
          m, kEndEffectorForceSensorName, kEndEffectorTorqueSensorName,
          kEndEffectorSensorSiteName, ee_force_sensor_id, ee_torque_sensor_id,
          ee_sensor_site_id)) {
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
  init_redis(redis_client, START_POS, START_ORIENTATION);

  //resetting the robot to the home position
  reset_to_home();

  //Creating all the opensai related objects and controllers --------------------
  robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
  std::cout << "Robot DOF: " << robot->dof() << "\n";
  std::cout << "MJ DOF: " << m->nq << "\n"; 
  control_point = Vector3d(0.0, 0.0, 0.05);
  control_frame = Affine3d::Identity();
  control_frame.translation() = control_point;
  motion_force_task = std::make_shared<SaiPrimitives::MotionForceTask>(robot, control_link, control_frame);
  motion_force_task->setPosControlGains(200, 20);
  
  motion_force_task->disableInternalOtg();
  joint_task = std::make_shared<SaiPrimitives::JointTask>(robot);
  update_robot_state(m, d);
  motion_force_task->reInitializeTask();
  joint_task->reInitializeTask();
  // ----------------------------------------------------------------------------

  // ---------------------------------FSPF------------------------------
  init_particle_filter(fs::path("particle_filter/pfilter_settings.yaml"),
                       force_space_particle_filter, force_dimension_queue);
  // -------------------------------------------------------------------

  if (is_data_collection) {
    try {
      init_haptic_controller(redis_client, robot, control_link,
                             haptic_controller, directions_of_proxy_feedback,
                             prev_sensed_force);
    } catch (const std::exception& exception) {
      std::cerr << exception.what() << "\n";
      mj_deleteData(d);
      mj_deleteModel(m);
      return 1;
    }
  }

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
