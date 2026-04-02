#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include "SaiModel.h"
#include "SaiPrimitives.h"

#include "redis/RedisClient.h"
#include "redis_keys.h"

SaiCommon::RedisClient redis_client;

using std::string;
namespace fs = std::filesystem;

namespace {

constexpr int kWindowWidth = 1200;
constexpr int kWindowHeight = 900;
constexpr int kSceneMaxGeometry = 2000;
constexpr mjtNum kRenderTimestep = 1.0 / 60.0;
constexpr int kRobotDof = 7;

mjModel* m = nullptr;
mjData* d = nullptr;
mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0.0;
double lasty = 0.0;

// Initialization variables for robot control ---------------------

Vector3d START_POS = Vector3d(0, 0.3, 0.5);
Matrix3d START_ORIENTATION = (Matrix3d() << 
    1,  0,  0,
    0, -1,  0,
    0,  0, -1).finished();

std::shared_ptr<SaiModel::SaiModel> robot;
std::shared_ptr<SaiPrimitives::MotionForceTask> motion_force_task;
std::shared_ptr<SaiPrimitives::JointTask> joint_task;

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

string robots_directory() {
  return (fs::path(FORCEWM_MODEL_ROOT) / "robots").string();
}

string resolve_model_path(const char* argument, const string& default_filename) {
  if (!argument) {
    return (fs::path(robots_directory()) / default_filename).string();
  }

  const fs::path input_path(argument);
  if (input_path.is_absolute() || input_path.has_parent_path()) {
    return input_path.string();
  }

  return (fs::path(robots_directory()) / input_path).string();
}

void print_usage(const char* executable_name) {
  std::cout << "Usage: " << executable_name
            << " [mujoco_xml] [robot_urdf]\n"
            << "Examples:\n"
            << "  " << executable_name << " fr3.xml fr3.urdf\n"
            << "  " << executable_name
            << " /full/path/to/fr3.xml /full/path/to/fr3.urdf\n";
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
void update_redis();
void query_redis_for_desired_state();

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

void controller_callback(const mjModel* m, mjData* d) {
  if (!robot || !motion_force_task || !joint_task) {
    return;
  }

  VectorXd qpos(kRobotDof);
  for (int i = 0; i < kRobotDof; ++i) {
    qpos(i) = d->qpos[i];
  }
  redis_client.setEigen(QPOS, qpos);

  update_robot_state(m, d);
  update_redis();
  query_redis_for_desired_state();
  motion_force_task->updateTaskModel(MatrixXd::Identity(robot->dof(), robot->dof()));
  joint_task->updateTaskModel(motion_force_task->getTaskAndPreviousNullspace());
  const VectorXd control_torques =
      motion_force_task->computeTorques() + joint_task->computeTorques() +
      robot->jointGravityVector();

  for (int i = 0; i < std::min<int>(control_torques.size(), m->nu); ++i) {
    d->ctrl[i] = control_torques(i);
  }
}

// -------- Redis Code ---------------------------------

void init_redis() {
    redis_client.setEigen(DESIRED_CARTESIAN_POSITION, START_POS);
    redis_client.setEigen(DESIRED_CARTESIAN_ORIENTATION, START_ORIENTATION);
    redis_client.setBool(RESET, false);
}

void update_redis() {
    Vector3d currentPosition = motion_force_task->getCurrentPosition();
    Matrix3d currentOrientation = motion_force_task->getCurrentOrientation();

    redis_client.setEigen(CURRENT_CARTESIAN_POSITION, currentPosition);
    redis_client.setEigen(CURRENT_CARTESIAN_ORIENTATION, currentOrientation);
}

void query_redis_for_desired_state() {
    const MatrixXd desired_position = redis_client.getEigen(DESIRED_CARTESIAN_POSITION);
    const MatrixXd desired_orientation = redis_client.getEigen(DESIRED_CARTESIAN_ORIENTATION);

    motion_force_task->setGoalPosition(desired_position.col(0).head<3>());
    motion_force_task->setGoalOrientation(desired_orientation.topLeftCorner<3, 3>());
}

// -------- Redis Code ---------------------------------

int main(int argc, char** argv) {
  if (argc > 3) {
    print_usage(argv[0]);
    return 1;
  }

  const string mujoco_file =
      resolve_model_path(argc > 1 ? argv[1] : nullptr, "fr3.xml");
  const string robot_file =
      resolve_model_path(argc > 2 ? argv[2] : nullptr, "fr3.urdf");

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

  robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
  std::cout << "Robot DOF: " << robot->dof() << "\n";
  std::cout << "MJ DOF: " << m->nq << "\n"; 
  std::string control_link = "fr3_link7";
  Vector3d control_point = Vector3d(0.0, 0.0, 0.0);
  Affine3d control_frame = Affine3d::Identity();
  control_frame.translation() = control_point;
  motion_force_task = std::make_shared<SaiPrimitives::MotionForceTask>(robot, control_link, control_frame);
  motion_force_task->disableInternalOtg();
  joint_task = std::make_shared<SaiPrimitives::JointTask>(robot);
  update_robot_state(m, d);
  motion_force_task->reInitializeTask();
  joint_task->reInitializeTask();

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
  mjr_defaultContext(&con);

  initialize_camera();
  mjv_makeScene(m, &scn, kSceneMaxGeometry);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  glfwSetKeyCallback(window, keyboard);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetScrollCallback(window, scroll);

  std::cout << "Viewer controls: left drag = rotate, right drag = pan, "
               "scroll = zoom, Backspace = reset, Esc = quit.\n";

  while (!glfwWindowShouldClose(window)) {
    const mjtNum simstart = d->time;

    while (d->time - simstart < kRenderTimestep) {
      mj_step(m, d);
    }

    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  mjv_freeScene(&scn);
  mjr_freeContext(&con);
  mjcb_control = nullptr;
  mj_deleteData(d);
  mj_deleteModel(m);

#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif

  return 0;
}
