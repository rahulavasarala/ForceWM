#pragma once

#include <filesystem>
#include <memory>
#include <queue>
#include <string>
#include <atomic>

#include <mujoco/mujoco.h>

#include "SaiModel.h"
#include "SaiPrimitives.h"
#include "particle_filter/ForceSpaceParticleFilter.h"
#include "redis/RedisClient.h"

bool initialize_force_sensor_handles(const mjModel* model,
                                     const char* force_sensor_name,
                                     const char* torque_sensor_name,
                                     const char* sensor_site_name,
                                     int& ee_force_sensor_id,
                                     int& ee_torque_sensor_id,
                                     int& ee_sensor_site_id);

Matrix3d get_force_sensor_rotation_in_world(const mjData* d,
                                            int ee_sensor_site_id);

Vector3d get_sensed_force(const mjModel* m,
                          const mjData* d,
                          int ee_force_sensor_id,
                          int ee_sensor_site_id,
                          bool express_in_world_frame = false);

Vector3d get_sensed_moment(const mjModel* m,
                           const mjData* d,
                           int ee_torque_sensor_id,
                           int ee_sensor_site_id,
                           bool express_in_world_frame = false);

void update_filtered_sensed_wrench(const mjModel* m,
                                   const mjData* d,
                                   int ee_force_sensor_id,
                                   int ee_torque_sensor_id,
                                   int ee_sensor_site_id,
                                   double low_pass_alpha,
                                   Vector3d& filtered_sensed_force_sensor_frame,
                                   Vector3d& filtered_sensed_moment_sensor_frame,
                                   bool& sensed_wrench_filter_initialized);

Vector3d get_filtered_sensed_force(
    const mjModel* m,
    const mjData* d,
    int ee_force_sensor_id,
    int ee_sensor_site_id,
    const Vector3d& filtered_sensed_force_sensor_frame,
    bool sensed_wrench_filter_initialized,
    bool express_in_world_frame = false);

Vector3d get_filtered_sensed_moment(
    const mjModel* m,
    const mjData* d,
    int ee_torque_sensor_id,
    int ee_sensor_site_id,
    const Vector3d& filtered_sensed_moment_sensor_frame,
    bool sensed_wrench_filter_initialized,
    bool express_in_world_frame = false);

void update_particle_filter(
    const std::shared_ptr<SaiPrimitives::MotionForceTask>& motion_force_task,
    const std::shared_ptr<ForceWM::ForceSpaceParticleFilter>&
        force_space_particle_filter,
    std::queue<int>& force_dimension_queue,
    Matrix3d& sigma_force,
    Matrix3d& sigma_motion,
    const mjModel* m,
    const mjData* d,
    int ee_force_sensor_id,
    int ee_sensor_site_id,
    const Vector3d& filtered_sensed_force_sensor_frame,
    bool sensed_wrench_filter_initialized);

void init_redis(SaiCommon::RedisClient& redis_client,
                const Vector3d& start_pos,
                const Matrix3d& start_orientation);

void update_reset_request_from_redis(SaiCommon::RedisClient& redis_client,
                                     std::atomic<bool>& reset_requested);

void update_redis(
    SaiCommon::RedisClient& redis_client,
    const std::shared_ptr<SaiPrimitives::MotionForceTask>& motion_force_task,
    const std::shared_ptr<ForceWM::ForceSpaceParticleFilter>&
        force_space_particle_filter,
    const mjModel* m,
    const mjData* d,
    int robot_dof,
    int ee_force_sensor_id,
    int ee_torque_sensor_id,
    int ee_sensor_site_id,
    const Vector3d& filtered_sensed_force_sensor_frame,
    const Vector3d& filtered_sensed_moment_sensor_frame,
    bool sensed_wrench_filter_initialized,
    Vector3d& force_or_motion_axis,
    int& force_dimension);

void query_redis_for_desired_state(
    SaiCommon::RedisClient& redis_client,
    const std::shared_ptr<SaiPrimitives::MotionForceTask>& motion_force_task,
    int force_dimension,
    const Vector3d& force_or_motion_axis);

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
    SaiPrimitives::HapticControllerInput& haptic_input);

void send_haptic_commands(
    SaiCommon::RedisClient& redis_client,
    const SaiPrimitives::HapticControllerOutput& haptic_output);

void init_haptic_controller(
    SaiCommon::RedisClient& redis_client,
    const std::shared_ptr<SaiModel::SaiModel>& robot,
    const std::string& control_link,
    std::shared_ptr<SaiPrimitives::HapticDeviceController>& haptic_controller,
    Vector3i& directions_of_proxy_feedback,
    Vector3d& prev_sensed_force);

void init_particle_filter(
    const std::filesystem::path& settings_path,
    std::shared_ptr<ForceWM::ForceSpaceParticleFilter>&
        force_space_particle_filter,
    std::queue<int>& force_dimension_queue);
