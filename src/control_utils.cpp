#include "control_utils.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include "redis/keys/chai_haptic_devices_driver.h"
#include "redis_keys.h"

using namespace SaiCommon::ChaiHapticDriverKeys;

namespace {

bool all_elements_same(std::queue<int> queue) {
  if (queue.empty()) {
    return true;
  }

  const int first = queue.front();
  while (!queue.empty()) {
    if (queue.front() != first) {
      return false;
    }
    queue.pop();
  }
  return true;
}

std::string queue_to_string(std::queue<int> queue) {
  std::ostringstream stream;
  stream << "[";
  bool first = true;
  while (!queue.empty()) {
    if (!first) {
      stream << ", ";
    }
    first = false;
    stream << queue.front();
    queue.pop();
  }
  stream << "]";
  return stream.str();
}

}  // namespace

bool initialize_force_sensor_handles(const mjModel* model,
                                     const char* force_sensor_name,
                                     const char* torque_sensor_name,
                                     const char* sensor_site_name,
                                     int& ee_force_sensor_id,
                                     int& ee_torque_sensor_id,
                                     int& ee_sensor_site_id) {
  ee_force_sensor_id = mj_name2id(model, mjOBJ_SENSOR, force_sensor_name);
  ee_torque_sensor_id = mj_name2id(model, mjOBJ_SENSOR, torque_sensor_name);
  ee_sensor_site_id = mj_name2id(model, mjOBJ_SITE, sensor_site_name);

  if (ee_force_sensor_id < 0 || ee_torque_sensor_id < 0 ||
      ee_sensor_site_id < 0) {
    std::cerr << "Could not find MuJoCo force/torque sensors `"
              << force_sensor_name << "` and `" << torque_sensor_name
              << "` or site `" << sensor_site_name
              << "` in the loaded model.\n";
    return false;
  }

  if (model->sensor_dim[ee_force_sensor_id] < 3 ||
      model->sensor_dim[ee_torque_sensor_id] < 3) {
    std::cerr << "MuJoCo force/torque sensors must each provide 3 values.\n";
    return false;
  }

  return true;
}

Matrix3d get_force_sensor_rotation_in_world(const mjData* d,
                                            int ee_sensor_site_id) {
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

Vector3d get_sensed_force(const mjModel* m,
                          const mjData* d,
                          int ee_force_sensor_id,
                          int ee_sensor_site_id,
                          const bool express_in_world_frame) {
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

  return get_force_sensor_rotation_in_world(d, ee_sensor_site_id) *
         sensed_force_sensor_frame;
}

Vector3d get_sensed_moment(const mjModel* m,
                           const mjData* d,
                           int ee_torque_sensor_id,
                           int ee_sensor_site_id,
                           const bool express_in_world_frame) {
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

  return get_force_sensor_rotation_in_world(d, ee_sensor_site_id) *
         sensed_moment_sensor_frame;
}

void update_filtered_sensed_wrench(const mjModel* m,
                                   const mjData* d,
                                   int ee_force_sensor_id,
                                   int ee_torque_sensor_id,
                                   int ee_sensor_site_id,
                                   double low_pass_alpha,
                                   Vector3d& filtered_sensed_force_sensor_frame,
                                   Vector3d& filtered_sensed_moment_sensor_frame,
                                   bool& sensed_wrench_filter_initialized) {
  const Vector3d raw_force_sensor_frame =
      get_sensed_force(m, d, ee_force_sensor_id, ee_sensor_site_id, false);
  const Vector3d raw_moment_sensor_frame =
      get_sensed_moment(m, d, ee_torque_sensor_id, ee_sensor_site_id, false);

  if (!sensed_wrench_filter_initialized) {
    filtered_sensed_force_sensor_frame = raw_force_sensor_frame;
    filtered_sensed_moment_sensor_frame = raw_moment_sensor_frame;
    sensed_wrench_filter_initialized = true;
    return;
  }

  filtered_sensed_force_sensor_frame =
      (1.0 - low_pass_alpha) * filtered_sensed_force_sensor_frame +
      low_pass_alpha * raw_force_sensor_frame;
  filtered_sensed_moment_sensor_frame =
      (1.0 - low_pass_alpha) * filtered_sensed_moment_sensor_frame +
      low_pass_alpha * raw_moment_sensor_frame;
}

Vector3d get_filtered_sensed_force(
    const mjModel* m,
    const mjData* d,
    int ee_force_sensor_id,
    int ee_sensor_site_id,
    const Vector3d& filtered_sensed_force_sensor_frame,
    bool sensed_wrench_filter_initialized,
    const bool express_in_world_frame) {
  const Vector3d sensed_force_sensor_frame =
      sensed_wrench_filter_initialized
          ? filtered_sensed_force_sensor_frame
          : get_sensed_force(m, d, ee_force_sensor_id, ee_sensor_site_id,
                             false);
  if (!express_in_world_frame) {
    return sensed_force_sensor_frame;
  }
  return get_force_sensor_rotation_in_world(d, ee_sensor_site_id) *
         sensed_force_sensor_frame;
}

Vector3d get_filtered_sensed_moment(
    const mjModel* m,
    const mjData* d,
    int ee_torque_sensor_id,
    int ee_sensor_site_id,
    const Vector3d& filtered_sensed_moment_sensor_frame,
    bool sensed_wrench_filter_initialized,
    const bool express_in_world_frame) {
  const Vector3d sensed_moment_sensor_frame =
      sensed_wrench_filter_initialized
          ? filtered_sensed_moment_sensor_frame
          : get_sensed_moment(m, d, ee_torque_sensor_id, ee_sensor_site_id,
                              false);
  if (!express_in_world_frame) {
    return sensed_moment_sensor_frame;
  }
  return get_force_sensor_rotation_in_world(d, ee_sensor_site_id) *
         sensed_moment_sensor_frame;
}

void update_particle_filter(
    const std::shared_ptr<SaiPrimitives::MotionForceTask>& motion_force_task,
    const std::shared_ptr<ForceWM::ForceSpaceParticleFilter>&
        force_space_particle_filter,
    PFilterOutput& pfilter_output,
    std::queue<int>& force_dimension_queue,
    Matrix3d& sigma_force,
    Matrix3d& sigma_motion,
    const mjModel* m,
    const mjData* d,
    int ee_force_sensor_id,
    int ee_sensor_site_id,
    const Vector3d& filtered_sensed_force_sensor_frame,
    bool sensed_wrench_filter_initialized) {
  Vector3d dx_world = motion_force_task->getGoalPosition() - motion_force_task->getCurrentPosition();

  const Vector3d robot_velocity = motion_force_task->getCurrentLinearVelocity();
  const Vector3d sensed_force_world_frame = get_filtered_sensed_force(
      m, d, ee_force_sensor_id, ee_sensor_site_id,
      filtered_sensed_force_sensor_frame, sensed_wrench_filter_initialized,
      true);

  const Vector3d motion_control = 50 * sigma_motion * dx_world;
  const Vector3d force_control = 50 * sigma_force * dx_world;

  force_space_particle_filter->update(motion_control, force_control,
                                      robot_velocity,
                                      sensed_force_world_frame);
  const Vector3d motion_or_force_axis =
      force_space_particle_filter->getForceOrMotionAxis();
  const int fdim = force_space_particle_filter->getForceSpaceDimension();
  force_dimension_queue.pop();
  force_dimension_queue.push(fdim);
  pfilter_output.flag_force_to_free = false;
  const bool queue_is_uniform = all_elements_same(force_dimension_queue);

  if (fdim == 0) {
    if (!queue_is_uniform) {
      std::cout << "Particle filter update: raw_fdim=" << fdim
                << " queue=" << queue_to_string(force_dimension_queue)
                << " force_to_free=false"
                << " action=hold_previous_constraint" << std::endl;
      return;
    }
    pfilter_output.flag_force_to_free = true;
  }

  pfilter_output.force_space_dimension = fdim;
  pfilter_output.force_or_motion_axis = motion_or_force_axis;
  std::cout << "Particle filter update: raw_fdim=" << fdim
            << " queue=" << queue_to_string(force_dimension_queue)
            << " force_to_free="
            << (pfilter_output.flag_force_to_free ? "true" : "false")
            << " committed_fdim=" << pfilter_output.force_space_dimension
            << "dx_world=" << dx_world.transpose()
            << "motion_control=" << motion_control.transpose()
            << "force_control=" << force_control.transpose()
            << std::endl;

  if (pfilter_output.force_space_dimension == 0) {
    sigma_force = Matrix3d::Zero();
    sigma_motion = Matrix3d::Identity();
  } else if (pfilter_output.force_space_dimension == 1) {
    sigma_force = pfilter_output.force_or_motion_axis *
                  pfilter_output.force_or_motion_axis.transpose();
    sigma_motion = Matrix3d::Identity() - sigma_force;
  } else if (pfilter_output.force_space_dimension == 2) {
    sigma_motion = pfilter_output.force_or_motion_axis *
                   pfilter_output.force_or_motion_axis.transpose();
    sigma_force = Matrix3d::Identity() - sigma_motion;
  } else if (pfilter_output.force_space_dimension == 3) {
    sigma_force = Matrix3d::Identity();
    sigma_motion = Matrix3d::Zero();
  }
}

void init_redis(SaiCommon::RedisClient& redis_client,
                const Vector3d& start_pos,
                const Matrix3d& start_orientation) {
  redis_client.setEigen(DESIRED_CARTESIAN_POSITION, start_pos);
  redis_client.setEigen(DESIRED_CARTESIAN_ORIENTATION, start_orientation);
  redis_client.setEigen(DESIRED_FORCE, Vector3d::Zero());
  redis_client.setEigen(FORCE_OR_MOTION_AXIS, Vector3d::Zero());
  redis_client.setInt(FORCE_DIMENSION, 0);
  redis_client.setBool(RESET, false);
}

void update_reset_request_from_redis(SaiCommon::RedisClient& redis_client,
                                     std::atomic<bool>& reset_requested) {
  if (!redis_client.getBool(RESET)) {
    return;
  }

  reset_requested = true;
  redis_client.setBool(RESET, false);
}

void update_redis(
    SaiCommon::RedisClient& redis_client,
    const std::shared_ptr<SaiPrimitives::MotionForceTask>& motion_force_task,
    const std::shared_ptr<ForceWM::ForceSpaceParticleFilter>&
        force_space_particle_filter,
    const PFilterOutput& pfilter_output,
    const mjModel* m,
    const mjData* d,
    int robot_dof,
    int ee_force_sensor_id,
    int ee_torque_sensor_id,
    int ee_sensor_site_id,
    const Vector3d& filtered_sensed_force_sensor_frame,
    const Vector3d& filtered_sensed_moment_sensor_frame,
    bool sensed_wrench_filter_initialized) {
  const Vector3d current_position = motion_force_task->getCurrentPosition();
  const Matrix3d current_orientation =
      motion_force_task->getCurrentOrientation();
  const Vector3d current_linear_velocity =
      motion_force_task->getCurrentLinearVelocity();

  redis_client.setEigen(CURRENT_CARTESIAN_POSITION, current_position);
  redis_client.setEigen(CURRENT_CARTESIAN_ORIENTATION, current_orientation);
  redis_client.setEigen(CURRENT_CARTESIAN_VELOCITY, current_linear_velocity);

  VectorXd qpos(robot_dof);
  for (int i = 0; i < robot_dof; ++i) {
    qpos(i) = d->qpos[i];
  }
  redis_client.setEigen(QPOS, qpos);

  const Vector3d sensed_force = -1 * get_filtered_sensed_force(
                                         m, d, ee_force_sensor_id,
                                         ee_sensor_site_id,
                                         filtered_sensed_force_sensor_frame,
                                         sensed_wrench_filter_initialized,
                                         true);
  const Vector3d sensed_moment = -1 * get_filtered_sensed_moment(
                                          m, d, ee_torque_sensor_id,
                                          ee_sensor_site_id,
                                          filtered_sensed_moment_sensor_frame,
                                          sensed_wrench_filter_initialized,
                                          true);
  redis_client.setEigen(SENSED_FORCE, sensed_force);
  redis_client.setEigen(SENSED_MOMENT, sensed_moment);

  (void)force_space_particle_filter;
  redis_client.setEigen(FORCE_OR_MOTION_AXIS, pfilter_output.force_or_motion_axis);
  redis_client.setInt(FORCE_DIMENSION, pfilter_output.force_space_dimension);
}

void query_redis_for_desired_state(
    SaiCommon::RedisClient& redis_client,
    const std::shared_ptr<SaiPrimitives::MotionForceTask>& motion_force_task,
    int force_dimension,
    const Vector3d& force_or_motion_axis) {
  const MatrixXd desired_position =
      redis_client.getEigen(DESIRED_CARTESIAN_POSITION);
  const MatrixXd desired_orientation =
      redis_client.getEigen(DESIRED_CARTESIAN_ORIENTATION);
  const Vector3d desired_force = redis_client.getEigen(DESIRED_FORCE);

  // motion_force_task->setGoalForce(desired_force);
  motion_force_task->parametrizeForceMotionSpaces(force_dimension,
                                                  force_or_motion_axis);
  motion_force_task->setGoalPosition(desired_position.col(0).head<3>());
  motion_force_task->setGoalOrientation(
      desired_orientation.topLeftCorner<3, 3>());
}

void update_haptic_information(
    const std::shared_ptr<SaiModel::SaiModel>& robot,
    const std::string& control_link,
    SaiCommon::RedisClient& redis_client,
    const mjModel* m,
    const mjData* d,
    int ee_force_sensor_id,
    int ee_torque_sensor_id,
    int ee_sensor_site_id,
    const Vector3d& filtered_sensed_force_sensor_frame,
    const Vector3d& filtered_sensed_moment_sensor_frame,
    bool sensed_wrench_filter_initialized,
    SaiPrimitives::HapticControllerInput& haptic_input) {
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
  haptic_input.robot_sensed_force =
      -1 * get_filtered_sensed_force(
               m, d, ee_force_sensor_id, ee_sensor_site_id,
               filtered_sensed_force_sensor_frame,
               sensed_wrench_filter_initialized, true);
  haptic_input.robot_sensed_moment =
      -1 * get_filtered_sensed_moment(
               m, d, ee_torque_sensor_id, ee_sensor_site_id,
               filtered_sensed_moment_sensor_frame,
               sensed_wrench_filter_initialized, true);
}

void send_haptic_commands(
    SaiCommon::RedisClient& redis_client,
    const SaiPrimitives::HapticControllerOutput& haptic_output) {
  redis_client.setEigen(createRedisKey(COMMANDED_FORCE_KEY_SUFFIX, 0),
                        haptic_output.device_command_force);
  redis_client.setEigen(createRedisKey(COMMANDED_TORQUE_KEY_SUFFIX, 0),
                        haptic_output.device_command_moment);
  redis_client.setEigen(HAPTIC_COMMANDED_POSITION,
                        haptic_output.robot_goal_position);
  redis_client.setEigen(HAPTIC_COMMANDED_ORIENTATION,
                        haptic_output.robot_goal_orientation);
}

void init_haptic_controller(
    SaiCommon::RedisClient& redis_client,
    const std::shared_ptr<SaiModel::SaiModel>& robot,
    const std::string& control_link,
    std::shared_ptr<SaiPrimitives::HapticDeviceController>& haptic_controller,
    Vector3i& directions_of_proxy_feedback,
    Vector3d& prev_sensed_force) {
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
      0.02 * device_limits.max_linear_stiffness,
      0.02 * device_limits.max_linear_damping,
      0.02 * device_limits.max_angular_stiffness,
      0.02 * device_limits.max_angular_damping);

  haptic_controller->setReductionFactorForce(0.05);
  haptic_controller->setReductionFactorMoment(0.05);
}

void init_particle_filter(
    const std::filesystem::path& settings_path,
    std::shared_ptr<ForceWM::ForceSpaceParticleFilter>&
        force_space_particle_filter,
    std::queue<int>& force_dimension_queue) {
  if (!std::filesystem::exists(settings_path)) {
    throw std::runtime_error("Could not find pfilter_settings.yaml");
  }

  const YAML::Node config = YAML::LoadFile(settings_path.string());
  if (!config["filter"]) {
    throw std::runtime_error("Missing 'filter' section in pfilter_settings.yaml");
  }

  const YAML::Node filter = config["filter"];

  const int n_particles = filter["n_particles"].as<int>();
  const double filter_freq = filter["filter_freq"].as<double>();
  const int queue_size = filter["queue_size"].as<int>();
  const double f_low = filter["f_low"].as<double>();
  const double f_high = filter["f_high"].as<double>();
  const double v_low = filter["v_low"].as<double>();
  const double v_high = filter["v_high"].as<double>();
  const double f_low_add = filter["f_low_add"].as<double>();
  const double f_high_add = filter["f_high_add"].as<double>();
  const double v_low_add = filter["v_low_add"].as<double>();
  const double v_high_add = filter["v_high_add"].as<double>();

  (void)filter_freq;

  std::cout << "Loaded particle filter settings from: " << settings_path
            << std::endl;
  std::cout << "Particle filter config: n_particles=" << n_particles
            << " queue_size=" << queue_size << std::endl;

  force_space_particle_filter =
      std::make_shared<ForceWM::ForceSpaceParticleFilter>(n_particles);
  force_space_particle_filter->setParameters(0, 0.025, 0.3, 0.05);
  force_space_particle_filter->setWeightingParameters(
      f_low, f_high, v_low, v_high, f_low_add, f_high_add, v_low_add,
      v_high_add);

  while (!force_dimension_queue.empty()) {
    force_dimension_queue.pop();
  }

  for (int i = 0; i < queue_size; ++i) {
    force_dimension_queue.push(0);
  }

  force_space_particle_filter->reset();
}
