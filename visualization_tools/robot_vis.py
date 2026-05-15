from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np


DEFAULT_SITE_NAME = "ft_site"
DEFAULT_KEYPOINT_GEOM = "tool_keypoint"
DEFAULT_X_AXIS_GEOM = "x_axis"
DEFAULT_Y_AXIS_GEOM = "y_axis"
DEFAULT_Z_AXIS_GEOM = "z_axis"
DEFAULT_KEYFRAME_NAME = "home"
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900


def _print_runtime_summary(
    wall_seconds: float,
    simulated_seconds: float,
    physics_step_count: int,
    rendered_frame_count: int,
) -> None:
    print("\nSimulation summary")
    print(f"  Wall time: {wall_seconds:.3f} s")
    print(f"  Simulated time: {simulated_seconds:.3f} s")
    print(f"  Physics steps: {physics_step_count}")
    print(f"  Rendered frames: {rendered_frame_count}")

    if wall_seconds > 1e-9:
        print(f"  Real-time factor: {simulated_seconds / wall_seconds:.3f}x")
        print(f"  Physics step rate: {physics_step_count / wall_seconds:.3f} steps/s")
        print(f"  Render FPS: {rendered_frame_count / wall_seconds:.3f} frames/s")
    else:
        print("  Runtime too short to compute rates.")

    if rendered_frame_count > 0:
        print(
            "  Avg steps per frame: "
            f"{physics_step_count / float(rendered_frame_count):.3f}"
        )


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def _quat_from_z_axis(target_axis_world: np.ndarray) -> np.ndarray:
    target_z = _normalize(target_axis_world)

    helper_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(helper_axis, target_z)) > 0.9:
        helper_axis = np.array([0.0, 1.0, 0.0], dtype=float)

    x_axis = _normalize(np.cross(helper_axis, target_z))
    y_axis = np.cross(target_z, x_axis)

    rotation_matrix = np.column_stack((x_axis, y_axis, target_z))
    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, rotation_matrix.reshape(-1))
    return quat


class RobotVis:
    def __init__(
        self,
        scene_path: Path,
        site_name: str = DEFAULT_SITE_NAME,
        keypoint_geom_name: str = DEFAULT_KEYPOINT_GEOM,
        x_axis_geom_name: str = DEFAULT_X_AXIS_GEOM,
        y_axis_geom_name: str = DEFAULT_Y_AXIS_GEOM,
        z_axis_geom_name: str = DEFAULT_Z_AXIS_GEOM,
        keyframe_name: str | None = DEFAULT_KEYFRAME_NAME,
        joint_angles: list[float] | None = None,
        axis_length: float = 0.08,
        axis_radius: float = 0.002,
        keypoint_radius: float = 0.006,
    ) -> None:
        self.scene_path = Path(scene_path).expanduser().resolve()
        if not self.scene_path.is_file():
            raise FileNotFoundError(f"Scene XML not found: {self.scene_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)

        self.site_id = self._require_id(mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.keypoint_geom_id = self._require_id(
            mujoco.mjtObj.mjOBJ_GEOM, keypoint_geom_name
        )
        self.x_axis_geom_id = self._require_id(
            mujoco.mjtObj.mjOBJ_GEOM, x_axis_geom_name
        )
        self.y_axis_geom_id = self._require_id(
            mujoco.mjtObj.mjOBJ_GEOM, y_axis_geom_name
        )
        self.z_axis_geom_id = self._require_id(
            mujoco.mjtObj.mjOBJ_GEOM, z_axis_geom_name
        )

        self.axis_length = float(axis_length)
        self.axis_radius = float(axis_radius)
        self.keypoint_radius = float(keypoint_radius)
        self._site_position = np.zeros(3, dtype=float)
        self._site_rotation = np.eye(3, dtype=float)
        self._button_left = False
        self._button_middle = False
        self._button_right = False
        self._last_cursor_x = 0.0
        self._last_cursor_y = 0.0

        if self.axis_length <= 0.0:
            raise ValueError("Axis length must be positive.")
        if self.axis_radius <= 0.0:
            raise ValueError("Axis radius must be positive.")
        if self.keypoint_radius <= 0.0:
            raise ValueError("Keypoint radius must be positive.")

        self._set_static_pose(keyframe_name=keyframe_name, joint_angles=joint_angles)
        self._hide_xml_marker_geometries()
        self._update_visual_geometries()

    def _require_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(
                f"Could not find {obj_type.name.lower().replace('mjobj_', '')} "
                f"named `{name}` in {self.scene_path}."
            )
        return obj_id

    def _set_static_pose(
        self, keyframe_name: str | None, joint_angles: list[float] | None
    ) -> None:
        if joint_angles is not None and keyframe_name is not None:
            raise ValueError(
                "Provide either explicit joint angles or a keyframe name, not both."
            )

        if joint_angles is not None:
            self._set_pose_from_joint_angles(joint_angles)
            return

        if keyframe_name is not None:
            keyframe_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name
            )
            if keyframe_id < 0:
                raise ValueError(
                    f"Keyframe `{keyframe_name}` was not found in {self.scene_path}."
                )
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        elif self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)

        self.data.qvel[:] = 0.0
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _set_pose_from_joint_angles(self, joint_angles: list[float]) -> None:
        expected_dofs = int(self.model.nq)
        if len(joint_angles) != expected_dofs:
            raise ValueError(
                f"Expected {expected_dofs} joint angles for this scene, "
                f"received {len(joint_angles)}."
            )

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.asarray(joint_angles, dtype=float)
        self.data.qvel[:] = 0.0
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _update_visual_geometries(self) -> None:
        self._site_position = np.array(self.data.site_xpos[self.site_id], dtype=float)
        self._site_rotation = np.array(
            self.data.site_xmat[self.site_id], dtype=float
        ).reshape(
            3, 3
        )

        mujoco.mj_forward(self.model, self.data)

    def _hide_xml_marker_geometries(self) -> None:
        for geom_id in (
            self.keypoint_geom_id,
            self.x_axis_geom_id,
            self.y_axis_geom_id,
            self.z_axis_geom_id,
        ):
            self.model.geom_rgba[geom_id, 3] = 0.0

    def _append_marker_geometries(self, scene: mujoco.MjvScene) -> None:
        self._append_sphere_geom(
            scene=scene,
            position=self._site_position,
            radius=self.keypoint_radius,
            rgba=np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
        )
        self._append_axis_geom(
            scene=scene,
            origin=self._site_position,
            axis_world=self._site_rotation[:, 0],
            rgba=np.array([1.0, 0.15, 0.15, 1.0], dtype=np.float32),
        )
        self._append_axis_geom(
            scene=scene,
            origin=self._site_position,
            axis_world=self._site_rotation[:, 1],
            rgba=np.array([0.15, 1.0, 0.15, 1.0], dtype=np.float32),
        )
        self._append_axis_geom(
            scene=scene,
            origin=self._site_position,
            axis_world=self._site_rotation[:, 2],
            rgba=np.array([0.15, 0.45, 1.0, 1.0], dtype=np.float32),
        )

    def _append_sphere_geom(
        self,
        scene: mujoco.MjvScene,
        position: np.ndarray,
        radius: float,
        rgba: np.ndarray,
    ) -> None:
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("MuJoCo scene does not have enough space for marker geoms.")

        geom = scene.geoms[scene.ngeom]
        size = np.array([radius, 0.0, 0.0], dtype=np.float64)
        pos = np.asarray(position, dtype=np.float64)
        mat = np.eye(3, dtype=np.float64).reshape(-1)
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE.value,
            size,
            pos,
            mat,
            rgba,
        )
        scene.ngeom += 1

    def _append_axis_geom(
        self,
        scene: mujoco.MjvScene,
        origin: np.ndarray,
        axis_world: np.ndarray,
        rgba: np.ndarray,
    ) -> None:
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("MuJoCo scene does not have enough space for marker geoms.")

        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CYLINDER.value,
            np.array([self.axis_radius, 0.5 * self.axis_length, 0.0], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            rgba,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CYLINDER.value,
            self.axis_radius,
            np.asarray(origin, dtype=np.float64),
            np.asarray(origin, dtype=np.float64)
            + self.axis_length * _normalize(np.asarray(axis_world, dtype=np.float64)),
        )
        scene.ngeom += 1

    def launch(self) -> None:
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW.")

        window = None
        wall_start_time = time.perf_counter()
        sim_start_time = float(self.data.time)
        physics_step_count = 0
        rendered_frame_count = 0
        try:
            glfw.window_hint(glfw.SAMPLES, 4)
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
            window = glfw.create_window(
                WINDOW_WIDTH,
                WINDOW_HEIGHT,
                f"RobotVis - {self.scene_path.name}",
                None,
                None,
            )
            if window is None:
                raise RuntimeError("Failed to create GLFW window.")

            glfw.make_context_current(window)
            glfw.swap_interval(1)

            cam = mujoco.MjvCamera()
            opt = mujoco.MjvOption()
            mujoco.mjv_defaultFreeCamera(self.model, cam)
            mujoco.mjv_defaultOption(opt)
            scene = mujoco.MjvScene(self.model, maxgeom=10000)
            context = mujoco.MjrContext(
                self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value
            )

            site_position = np.array(self.data.site_xpos[self.site_id], dtype=float)
            cam.lookat[:] = site_position
            cam.distance = 0.7
            cam.azimuth = 135.0
            cam.elevation = -25.0

            self._install_callbacks(window, cam, scene)

            while not glfw.window_should_close(window):
                self.data.qvel[:] = 0.0
                if self.model.nu > 0:
                    self.data.ctrl[:] = 0.0
                mujoco.mj_forward(self.model, self.data)
                self._update_visual_geometries()

                width, height = glfw.get_framebuffer_size(window)
                viewport = mujoco.MjrRect(0, 0, width, height)
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    opt,
                    None,
                    cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    scene,
                )
                self._append_marker_geometries(scene)
                mujoco.mjr_render(viewport, scene, context)

                glfw.swap_buffers(window)
                glfw.poll_events()
                rendered_frame_count += 1
                time.sleep(1.0 / 120.0)
        finally:
            wall_seconds = time.perf_counter() - wall_start_time
            simulated_seconds = float(self.data.time) - sim_start_time
            if window is not None:
                glfw.destroy_window(window)
            glfw.terminate()
            _print_runtime_summary(
                wall_seconds=wall_seconds,
                simulated_seconds=simulated_seconds,
                physics_step_count=physics_step_count,
                rendered_frame_count=rendered_frame_count,
            )

    def _install_callbacks(
        self, window: glfw._GLFWwindow, cam: mujoco.MjvCamera, scene: mujoco.MjvScene
    ) -> None:
        glfw.set_window_user_pointer(window, self)

        def mouse_button_callback(
            callback_window: glfw._GLFWwindow, button: int, action: int, mods: int
        ) -> None:
            del mods
            self._button_left = (
                glfw.get_mouse_button(callback_window, glfw.MOUSE_BUTTON_LEFT)
                == glfw.PRESS
            )
            self._button_middle = (
                glfw.get_mouse_button(callback_window, glfw.MOUSE_BUTTON_MIDDLE)
                == glfw.PRESS
            )
            self._button_right = (
                glfw.get_mouse_button(callback_window, glfw.MOUSE_BUTTON_RIGHT)
                == glfw.PRESS
            )
            self._last_cursor_x, self._last_cursor_y = glfw.get_cursor_pos(
                callback_window
            )
            if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
                self._button_left = False
            if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.RELEASE:
                self._button_middle = False
            if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.RELEASE:
                self._button_right = False

        def cursor_pos_callback(
            callback_window: glfw._GLFWwindow, xpos: float, ypos: float
        ) -> None:
            if not (self._button_left or self._button_middle or self._button_right):
                self._last_cursor_x = xpos
                self._last_cursor_y = ypos
                return

            dx = xpos - self._last_cursor_x
            dy = ypos - self._last_cursor_y
            self._last_cursor_x = xpos
            self._last_cursor_y = ypos

            _, window_height = glfw.get_window_size(callback_window)
            window_height = max(window_height, 1)
            shift_pressed = (
                glfw.get_key(callback_window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                or glfw.get_key(callback_window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
            )

            if self._button_right:
                action = (
                    mujoco.mjtMouse.mjMOUSE_MOVE_H
                    if shift_pressed
                    else mujoco.mjtMouse.mjMOUSE_MOVE_V
                )
            elif self._button_left:
                action = (
                    mujoco.mjtMouse.mjMOUSE_ROTATE_H
                    if shift_pressed
                    else mujoco.mjtMouse.mjMOUSE_ROTATE_V
                )
            else:
                action = mujoco.mjtMouse.mjMOUSE_ZOOM

            mujoco.mjv_moveCamera(
                self.model,
                action,
                dx / window_height,
                dy / window_height,
                scene,
                cam,
            )

        def scroll_callback(
            callback_window: glfw._GLFWwindow, xoffset: float, yoffset: float
        ) -> None:
            del callback_window, xoffset
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_ZOOM,
                0.0,
                -0.05 * yoffset,
                scene,
                cam,
            )

        def key_callback(
            callback_window: glfw._GLFWwindow,
            key: int,
            scancode: int,
            action: int,
            mods: int,
        ) -> None:
            del scancode, mods
            if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
                glfw.set_window_should_close(callback_window, True)

        glfw.set_mouse_button_callback(window, mouse_button_callback)
        glfw.set_cursor_pos_callback(window, cursor_pos_callback)
        glfw.set_scroll_callback(window, scroll_callback)
        glfw.set_key_callback(window, key_callback)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the end-effector keypoint and frame axes for a MuJoCo scene."
        )
    )
    parser.add_argument(
        "scene_xml",
        type=Path,
        help="Path to the MuJoCo XML scene file.",
    )
    parser.add_argument(
        "--site",
        default=DEFAULT_SITE_NAME,
        help=f"Site used as the end-effector keypoint. Default: {DEFAULT_SITE_NAME}",
    )
    parser.add_argument(
        "--keypoint-geom",
        default=DEFAULT_KEYPOINT_GEOM,
        help=(
            "Name of the sphere geom that should be moved onto the site. "
            f"Default: {DEFAULT_KEYPOINT_GEOM}"
        ),
    )
    parser.add_argument(
        "--x-axis-geom",
        default=DEFAULT_X_AXIS_GEOM,
        help=f"Name of the x-axis cylinder geom. Default: {DEFAULT_X_AXIS_GEOM}",
    )
    parser.add_argument(
        "--y-axis-geom",
        default=DEFAULT_Y_AXIS_GEOM,
        help=f"Name of the y-axis cylinder geom. Default: {DEFAULT_Y_AXIS_GEOM}",
    )
    parser.add_argument(
        "--z-axis-geom",
        default=DEFAULT_Z_AXIS_GEOM,
        help=f"Name of the z-axis cylinder geom. Default: {DEFAULT_Z_AXIS_GEOM}",
    )
    parser.add_argument(
        "--keyframe",
        default=DEFAULT_KEYFRAME_NAME,
        help=(
            "Keyframe name used to hold the robot in a static pose. "
            "Pass an empty string together with --joint-angles to disable it."
        ),
    )
    parser.add_argument(
        "--joint-angles",
        type=float,
        nargs="+",
        help=(
            "Explicit joint angles to use instead of a keyframe. "
            "Must match the scene qpos dimension exactly."
        ),
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.08,
        help="Rendered axis length in meters. Default: 0.08",
    )
    parser.add_argument(
        "--axis-radius",
        type=float,
        default=0.002,
        help="Rendered axis cylinder radius in meters. Default: 0.002",
    )
    parser.add_argument(
        "--keypoint-radius",
        type=float,
        default=0.006,
        help="Rendered keypoint sphere radius in meters. Default: 0.006",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    keyframe_name = args.keyframe if args.keyframe else None
    if args.joint_angles is not None:
        keyframe_name = None

    robot_vis = RobotVis(
        scene_path=args.scene_xml,
        site_name=args.site,
        keypoint_geom_name=args.keypoint_geom,
        x_axis_geom_name=args.x_axis_geom,
        y_axis_geom_name=args.y_axis_geom,
        z_axis_geom_name=args.z_axis_geom,
        keyframe_name=keyframe_name,
        joint_angles=args.joint_angles,
        axis_length=args.axis_length,
        axis_radius=args.axis_radius,
        keypoint_radius=args.keypoint_radius,
    )
    robot_vis.launch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
