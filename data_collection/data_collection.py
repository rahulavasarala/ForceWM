from __future__ import annotations

import argparse
import os
import select
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

if __package__:
    from data_collection.camera_observer import CameraObserver
    from data_collection.robot_observer import RobotObserver
    from data_collection.saver import Saver
else:
    from camera_observer import CameraObserver
    from robot_observer import RobotObserver
    from saver import Saver


class DataCollection:
    DEFAULT_FPS = 30.0

    def __init__(self, save_dir: str, data_name: str, universal_contract: str):
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.data_name = data_name
        self.contract_path = Path(universal_contract).expanduser().resolve()
        self.buffer_dir = self.save_dir / self.data_name

        self.contract = self._load_contract(self.contract_path)
        self.lowdim_cfg = self._require_source_cfg("lowdim")
        self.visual_cfg = self._require_source_cfg("visual")

        self.lowdim_fps = self._resolve_fps(self.lowdim_cfg, "lowdim")
        self.camera_fps = self._resolve_fps(self.visual_cfg, "visual")
        self.lowdim_buffer_size = self._resolve_buffer_size(self.lowdim_cfg, "lowdim")
        self.camera_buffer_size = self._resolve_buffer_size(self.visual_cfg, "visual")

        self.robot_observer: RobotObserver | None = None
        self.camera_observer: CameraObserver | None = None
        self.saver: Saver | None = None

        self._keyboard_thread: threading.Thread | None = None
        self._saving_indicator_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._recording_event = threading.Event()
        self._recording_lock = threading.Lock()

        self._terminal_settings: Any = None
        self._stdin_fd: int | None = None

    def _launch_observers(self) -> None:
        if self.robot_observer is None:
            self.robot_observer = RobotObserver(
                buffer_size=self.lowdim_buffer_size,
                example_obs={},
                obs_freq=self.lowdim_fps,
                robot_data=self.contract_path,
            )

        if self.camera_observer is None:
            self.camera_observer = CameraObserver(
                buffer_size=self.camera_buffer_size,
                example_obs={},
                camera_freq=self.camera_fps,
                robot_data=self.contract_path,
            )

        self.robot_observer.start_adding_obs()
        self.camera_observer.start_adding_obs()

    def _launch_saver(self) -> None:
        if self.robot_observer is None or self.camera_observer is None:
            raise RuntimeError("Observers must be launched before the saver.")

        if self.saver is None:
            self.saver = Saver(
                save_dir=self.buffer_dir,
                robot_observer=self.robot_observer,
                camera_observer=self.camera_observer,
                contract_path=self.contract_path,
            )

    def _stop_observers(self) -> None:
        if self.camera_observer is not None:
            self.camera_observer.stop_adding_obs()

        if self.robot_observer is not None:
            self.robot_observer.stop_adding_obs()

    def _stop_saver(self) -> None:
        if self.saver is not None:
            self.saver.quit()

    def start_recording(self) -> None:
        if self.saver is None:
            raise RuntimeError("Saver is not initialized.")

        with self._recording_lock:
            if self._recording_event.is_set():
                print("Recording is already active.", flush=True)
                return

            start_timestamp_s = self.saver.start()
            self._recording_event.set()
            episode_name = self.saver.current_episode_name or "unknown_episode"
            episode_path = self.saver.current_episode_path
            if episode_path is not None:
                print(
                    f"Started recording {episode_name} at {start_timestamp_s:.3f}s -> {episode_path}",
                    flush=True,
                )
            else:
                print(f"Started recording {episode_name} at {start_timestamp_s:.3f}s.", flush=True)

    def stop_recording(self) -> None:
        if self.saver is None:
            raise RuntimeError("Saver is not initialized.")

        with self._recording_lock:
            if not self._recording_event.is_set():
                print("Recording is not currently active.", flush=True)
                return

            self.saver.stop()
            self._recording_event.clear()
            summary = self.saver.last_completed_episode_summary
            if summary is None:
                print("Stopped recording.", flush=True)
                return

            episode_name = summary.get("episode_name", "unknown_episode")
            episode_path = summary.get("episode_path", "unknown_path")
            print(f"Saved {episode_name} in {episode_path}", flush=True)
            print(self._format_save_summary(summary), flush=True)

    def run(self) -> None:
        self.buffer_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._launch_observers()
            self._launch_saver()
            self._start_keyboard_listener()
            self._start_saving_indicator()

            print(f"Data collection ready. Episodes will be saved under {self.buffer_dir}", flush=True)
            print("Press `k` to start recording, `l` to stop recording, Ctrl-C to exit.", flush=True)

            while not self._shutdown_event.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nShutting down data collection.", flush=True)
        finally:
            self._shutdown_event.set()

            if self._recording_event.is_set():
                try:
                    self.stop_recording()
                except Exception as exc:
                    print(f"Failed to stop the active recording cleanly: {exc}", flush=True)
                finally:
                    self._recording_event.clear()

            try:
                self._stop_saver()
            except Exception as exc:
                print(f"Failed to finalize saver cleanly: {exc}", flush=True)

            self._stop_observers()
            self._join_background_threads()
            self._restore_terminal_settings()

    def _start_keyboard_listener(self) -> None:
        if os.name != "posix":
            raise RuntimeError("The data collection keyboard listener currently supports POSIX terminals only.")
        if not sys.stdin.isatty():
            raise RuntimeError("data_collection.py must be run from a real terminal to capture `k`/`l` input.")
        if self._keyboard_thread is not None and self._keyboard_thread.is_alive():
            return

        import termios
        import tty

        self._stdin_fd = sys.stdin.fileno()
        if self._terminal_settings is None:
            self._terminal_settings = termios.tcgetattr(self._stdin_fd)

        tty.setcbreak(self._stdin_fd)
        self._keyboard_thread = threading.Thread(
            target=self._keyboard_listener_loop,
            name="data-collection-keyboard",
            daemon=True,
        )
        self._keyboard_thread.start()

    def _keyboard_listener_loop(self) -> None:
        while not self._shutdown_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue

            key = sys.stdin.read(1)
            if not key:
                continue

            try:
                if key.lower() == "k":
                    self.start_recording()
                elif key.lower() == "l":
                    self.stop_recording()
            except Exception as exc:
                print(f"Keyboard command failed: {exc}", flush=True)

    def _start_saving_indicator(self) -> None:
        if self._saving_indicator_thread is not None and self._saving_indicator_thread.is_alive():
            return

        self._saving_indicator_thread = threading.Thread(
            target=self._saving_indicator_loop,
            name="data-collection-indicator",
            daemon=True,
        )
        self._saving_indicator_thread.start()

    def _saving_indicator_loop(self) -> None:
        while not self._shutdown_event.is_set():
            if self._recording_event.is_set():
                print("saving...", flush=True)
                if self._shutdown_event.wait(1.0):
                    return
            else:
                if self._shutdown_event.wait(0.1):
                    return

    def _join_background_threads(self) -> None:
        if self._keyboard_thread is not None:
            self._keyboard_thread.join(timeout=1.0)

        if self._saving_indicator_thread is not None:
            self._saving_indicator_thread.join(timeout=1.0)

    def _restore_terminal_settings(self) -> None:
        if self._stdin_fd is None or self._terminal_settings is None:
            return

        import termios

        termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._terminal_settings)
        self._stdin_fd = None
        self._terminal_settings = None

    @staticmethod
    def _format_save_summary(summary: dict[str, Any]) -> str:
        duration_s = float(summary.get("duration_s", 0.0))
        num_lowdim_samples = int(summary.get("num_lowdim_samples", 0))
        camera_frame_counts = summary.get("camera_frame_counts", {})

        if isinstance(camera_frame_counts, dict) and camera_frame_counts:
            camera_summary = ", ".join(
                f"{camera_name}: {frame_count} frames"
                for camera_name, frame_count in sorted(camera_frame_counts.items())
            )
        else:
            camera_summary = "no camera frames"

        return (
            f"Summary: duration={duration_s:.2f}s, "
            f"lowdim_samples={num_lowdim_samples}, "
            f"cameras=[{camera_summary}]"
        )

    def _require_source_cfg(self, source_name: str) -> dict[str, Any]:
        robot_cfg = self.contract.get("robot")
        if not isinstance(robot_cfg, dict):
            raise ValueError("The universal contract must contain a top-level `robot` mapping.")

        data_sources = robot_cfg.get("data_sources")
        if not isinstance(data_sources, dict):
            raise ValueError("The universal contract must contain `robot.data_sources`.")

        source_cfg = data_sources.get(source_name)
        if not isinstance(source_cfg, dict):
            raise ValueError(f"The universal contract must contain `robot.data_sources.{source_name}`.")

        if not isinstance(source_cfg.get("keys"), list) or not source_cfg["keys"]:
            raise ValueError(
                f"`robot.data_sources.{source_name}.keys` must be a non-empty list in the universal contract."
            )

        return source_cfg

    def _resolve_fps(self, source_cfg: dict[str, Any], source_name: str) -> float:
        fps = float(source_cfg.get("fps", self.DEFAULT_FPS))
        if fps <= 0.0:
            raise ValueError(f"`robot.data_sources.{source_name}.fps` must be positive.")
        return fps

    def _resolve_buffer_size(self, source_cfg: dict[str, Any], source_name: str) -> int:
        max_obs_window = 0
        for key_entry in source_cfg["keys"]:
            if not isinstance(key_entry, dict) or len(key_entry) != 1:
                raise ValueError(
                    f"Each entry in `robot.data_sources.{source_name}.keys` must be a single-key mapping."
                )

            key_name, key_cfg = next(iter(key_entry.items()))
            if not isinstance(key_cfg, dict):
                raise ValueError(
                    f"`robot.data_sources.{source_name}.keys.{key_name}` must map to a dictionary."
                )

            obs_window = key_cfg.get("obs_window")
            if obs_window is None:
                raise ValueError(
                    f"`robot.data_sources.{source_name}.keys.{key_name}.obs_window` is required "
                    "to size the observer buffer."
                )

            max_obs_window = max(max_obs_window, int(obs_window))

        return max(1, 3 * max_obs_window)

    @staticmethod
    def _load_contract(contract_path: Path) -> dict[str, Any]:
        with contract_path.open("r", encoding="utf-8") as handle:
            contract = yaml.safe_load(handle)

        if not isinstance(contract, dict):
            raise ValueError("The universal contract must load as a dictionary.")

        return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive data collection from Redis-backed observers.")
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        required=True,
        type=str,
        help="Directory that contains the named recording buffers.",
    )
    parser.add_argument(
        "--buffer-name",
        dest="buffer_name",
        required=True,
        type=str,
        help="Name of the recording buffer directory.",
    )
    parser.add_argument(
        "--universal-contract",
        dest="universal_contract",
        required=True,
        type=str,
        help="Path to the universal contract file.",
    )
    args = parser.parse_args()

    DataCollection(args.save_dir, args.buffer_name, args.universal_contract).run()
