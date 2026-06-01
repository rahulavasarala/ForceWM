from __future__ import annotations

import json
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from data_collection.camera_observer import CameraObserver
from data_collection.robot_observer import RobotObserver
from data_collection.saver import Saver


def _json_bytes(payload) -> bytes:
    return json.dumps(payload).encode("utf-8")


class _FakeRedis:
    def __init__(self, values: dict[str, list[bytes] | bytes]) -> None:
        self._values = {}
        for key, value in values.items():
            if isinstance(value, list):
                self._values[key] = list(value)
            else:
                self._values[key] = [value]

    def get(self, key: str):
        values = self._values.get(key)
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return values.pop(0)

    def ping(self) -> bool:
        return True


class _FakeWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(np.asarray(frame).copy())

    def release(self) -> None:
        return None


class _FakeObserver:
    def __init__(
        self,
        *,
        buffer: list[dict],
        contract: dict,
        lowdim_specs: dict | None = None,
        camera_specs: dict | None = None,
        obs_freq: float | None = None,
        camera_freq: float | None = None,
        source_type: str | None = None,
    ) -> None:
        self.buffer = deque(buffer, maxlen=128)
        self._lock = threading.Lock()
        self.contract = contract
        self.lowdim_specs = lowdim_specs
        self.camera_specs = camera_specs
        self.obs_freq = obs_freq
        self.camera_freq = camera_freq
        self.source_type = source_type

    def get_latest_obs(self) -> dict | None:
        with self._lock:
            if not self.buffer:
                return None
            return dict(self.buffer[-1])


class RobotObserverTimestampTests(unittest.TestCase):
    def test_sim_lowdim_uses_stable_metadata_sim_time(self) -> None:
        contract = {
            "robot": {
                "redis_namespace": "sai",
                "prefix": "sim::franka",
                "type": "sim",
                "data_sources": {
                    "lowdim": {
                        "fps": 30,
                        "keys": [
                            {"eef_pos": {"redis": "current_cartesian_position", "dim": [3]}},
                            {"eef_ori": {"redis": "current_cartesian_orientation", "dim": [3, 3]}},
                        ]
                    }
                },
            }
        }
        observer = RobotObserver(buffer_size=4, example_obs={}, obs_freq=30, robot_data=contract)
        metadata = _json_bytes(
            {
                "seq": 17,
                "sim_time_s": 2.5,
                "publish_wall_time_s": 101.0,
            }
        )
        observer.redis_client = _FakeRedis(
            {
                "sai::sim::franka::lowdim::meta": metadata,
                "sai::sim::franka::current_cartesian_position": _json_bytes([0.1, 0.2, 0.3]),
                "sai::sim::franka::current_cartesian_orientation": _json_bytes(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                ),
            }
        )

        observation = observer._read_observation_from_redis()

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation["source_seq"], 17)
        self.assertAlmostEqual(observation["timestamp_s"], 2.5)
        self.assertAlmostEqual(observation["source_publish_wall_time_s"], 101.0)

        with mock.patch("data_collection.robot_observer.time.time", return_value=200.0):
            finalized = observer._finalize_observation(observation)

        self.assertAlmostEqual(finalized["timestamp_s"], 2.5)
        self.assertAlmostEqual(finalized["observer_wall_time_s"], 200.0)
        self.assertEqual(finalized["observer_seq"], 0)

    def test_sim_lowdim_uses_first_metadata_snapshot_when_publish_advances(self) -> None:
        contract = {
            "robot": {
                "redis_namespace": "sai",
                "prefix": "sim::franka",
                "type": "sim",
                "data_sources": {
                    "lowdim": {
                        "fps": 30,
                        "keys": [{"eef_pos": {"redis": "current_cartesian_position", "dim": [3]}}],
                    }
                },
            }
        }
        observer = RobotObserver(buffer_size=4, example_obs={}, obs_freq=30, robot_data=contract)
        observer.redis_client = _FakeRedis(
            {
                "sai::sim::franka::lowdim::meta": [
                    _json_bytes(
                        {
                            "seq": 8,
                            "sim_time_s": 1.0,
                            "publish_wall_time_s": 10.0,
                        }
                    ),
                    _json_bytes(
                        {
                            "seq": 9,
                            "sim_time_s": 1.1,
                            "publish_wall_time_s": 10.1,
                        }
                    ),
                ],
                "sai::sim::franka::current_cartesian_position": _json_bytes([0.1, 0.2, 0.3]),
            }
        )

        observation = observer._read_observation_from_redis()

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation["source_seq"], 8)
        self.assertAlmostEqual(observation["timestamp_s"], 1.0)
        self.assertAlmostEqual(observation["source_publish_wall_time_s"], 10.0)


class CameraObserverTimestampTests(unittest.TestCase):
    def test_sim_camera_uses_stable_metadata_sim_time(self) -> None:
        ok, encoded = cv2.imencode(".jpg", np.zeros((2, 2, 3), dtype=np.uint8))
        self.assertTrue(ok)

        contract = {
            "robot": {
                "redis_namespace": "sai",
                "prefix": "sim::franka",
                "type": "sim",
                "data_sources": {
                    "visual": {
                        "fps": 30,
                        "keys": [{"camera_01": {"type": "rgb", "dim": [2, 2, 3]}}],
                    }
                },
            }
        }
        observer = CameraObserver(buffer_size=4, example_obs={}, camera_freq=30, robot_data=contract)
        metadata = _json_bytes(
            {
                "seq": 11,
                "sim_time_s": 4.25,
                "publish_wall_time_s": 202.0,
                "reset_epoch": 2,
                "is_complete": True,
            }
        )
        observer.redis_client = _FakeRedis(
            {
                "sai::sim::franka::camera_01::meta": metadata,
                "sai::sim::franka::camera_01": bytes(encoded),
            }
        )

        observation = observer._read_observation()

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertAlmostEqual(observation["timestamp_s"], 4.25)
        self.assertAlmostEqual(observation["sim_timestamp_s"], 4.25)
        self.assertEqual(observation["camera_frame_seq"], 11)
        self.assertAlmostEqual(observation["camera_source_timestamps"]["camera_01"], 202.0)

        with mock.patch("data_collection.camera_observer.time.time", return_value=300.0):
            finalized = observer._finalize_observation(observation)

        self.assertAlmostEqual(finalized["timestamp_s"], 4.25)
        self.assertAlmostEqual(finalized["observer_wall_time_s"], 300.0)
        self.assertEqual(finalized["observer_seq"], 0)

    def test_sim_camera_rejects_changed_metadata(self) -> None:
        ok, encoded = cv2.imencode(".jpg", np.zeros((2, 2, 3), dtype=np.uint8))
        self.assertTrue(ok)

        contract = {
            "robot": {
                "redis_namespace": "sai",
                "prefix": "sim::franka",
                "type": "sim",
                "data_sources": {
                    "visual": {
                        "fps": 30,
                        "keys": [{"camera_01": {"type": "rgb", "dim": [2, 2, 3]}}],
                    }
                },
            }
        }
        observer = CameraObserver(buffer_size=4, example_obs={}, camera_freq=30, robot_data=contract)
        observer.redis_client = _FakeRedis(
            {
                "sai::sim::franka::camera_01::meta": [
                    _json_bytes(
                        {
                            "seq": 4,
                            "sim_time_s": 1.0,
                            "publish_wall_time_s": 10.0,
                            "is_complete": True,
                        }
                    ),
                    _json_bytes(
                        {
                            "seq": 5,
                            "sim_time_s": 1.1,
                            "publish_wall_time_s": 10.1,
                            "is_complete": True,
                        }
                    ),
                ],
                "sai::sim::franka::camera_01": bytes(encoded),
            }
        )

        self.assertIsNone(observer._read_observation())

    def test_non_sim_camera_falls_back_to_observer_wall_time(self) -> None:
        contract = {
            "robot": {
                "redis_namespace": "sai",
                "prefix": "real::franka",
                "type": "real",
                "data_sources": {
                    "visual": {
                        "fps": 30,
                        "keys": [{"camera_01": {"type": "rgb", "dim": [2, 2, 3]}}],
                    }
                },
            }
        }
        observer = CameraObserver(buffer_size=4, example_obs={}, camera_freq=30, robot_data=contract)
        observer.redis_client = _FakeRedis(
            {
                "sai::real::franka::camera_01": _json_bytes(
                    [
                        [[0, 0, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0]],
                    ]
                )
            }
        )

        observation = observer._read_observation()
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertNotIn("timestamp_s", observation)

        with mock.patch("data_collection.camera_observer.time.time", return_value=42.5):
            finalized = observer._finalize_observation(observation)

        self.assertAlmostEqual(finalized["timestamp_s"], 42.5)
        self.assertAlmostEqual(finalized["observer_wall_time_s"], 42.5)


class SaverTimestampTests(unittest.TestCase):
    def _make_contract(self) -> dict:
        return {
            "robot": {
                "type": "sim",
                "data_sources": {
                    "lowdim": {
                        "fps": 30,
                        "keys": [
                            {"eef_pos": {"dim": [3]}},
                            {"eef_ori": {"dim": [3, 3]}},
                        ],
                    },
                    "visual": {
                        "fps": 30,
                        "keys": [{"camera_01": {"type": "rgb", "dim": [2, 2, 3]}}],
                    },
                },
            }
        }

    def _prepare_saver(
        self,
        tmp_dir: str,
        lowdim_buffer: list[dict],
        camera_buffer: list[dict],
        contract: dict | None = None,
        contract_path: Path | None = None,
    ) -> tuple[Saver, _FakeWriter]:
        contract = self._make_contract() if contract is None else contract
        robot_observer = _FakeObserver(
            buffer=lowdim_buffer,
            contract=contract,
            lowdim_specs={"eef_pos": {"dim": [3]}, "eef_ori": {"dim": [3, 3]}},
            obs_freq=30.0,
            source_type="sim",
        )
        camera_observer = _FakeObserver(
            buffer=camera_buffer,
            contract=contract,
            camera_specs={"camera_01": {"type": "rgb", "dim": [2, 2, 3], "fps": 30.0, "source_type": "sim"}},
            camera_freq=30.0,
        )
        saver = Saver(
            save_dir=Path(tmp_dir) / "episodes",
            robot_observer=robot_observer,
            camera_observer=camera_observer,
            contract_path=contract_path,
        )
        saver._episode_id = 1
        saver._episode_dir = saver.save_dir / "episode_000001"
        saver._visual_dir = saver._episode_dir / "visual"
        saver._episode_dir.mkdir(parents=True, exist_ok=False)
        saver._visual_dir.mkdir(parents=True, exist_ok=False)
        saver._start_timestamp_s = 100.0
        saver._reset_episode_storage()
        saver._seed_saved_markers()

        writer = _FakeWriter()
        saver._video_writers["camera_01"] = writer
        saver._video_paths["camera_01"] = saver._visual_dir / "camera_01.mp4"
        return saver, writer

    def test_saver_saves_latest_samples_once_and_skips_prestart_data(self) -> None:
        identity = np.eye(3, dtype=np.float64)
        lowdim_prestart = [
            {
                "observer_seq": 0,
                "timestamp_s": 3.0,
                "source_seq": 30,
                "eef_pos": np.array([0.0, 0.0, 0.0]),
                "eef_ori": identity,
            }
        ]
        camera_prestart = [
            {
                "observer_seq": 0,
                "timestamp_s": 2.0,
                "camera_frame_seq": 20,
                "camera_frame_seqs": {"camera_01": 20},
                "camera_01": np.zeros((2, 2, 3), dtype=np.uint8),
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            saver, writer = self._prepare_saver(tmp_dir, lowdim_prestart, camera_prestart)

            saver._tick_once()
            self.assertEqual(saver._lowdim_records["timestamp_s"], [])
            self.assertEqual(saver._camera_timestamps["camera_01"], [])

            with saver.robot_observer._lock:
                saver.robot_observer.buffer.extend(
                    [
                        {
                            "observer_seq": 1,
                            "timestamp_s": 3.0,
                            "source_seq": 30,
                            "eef_pos": np.array([0.0, 0.0, 0.0]),
                            "eef_ori": identity,
                        },
                        {
                            "observer_seq": 2,
                            "timestamp_s": 3.1,
                            "source_seq": 31,
                            "eef_pos": np.array([1.0, 0.0, 0.0]),
                            "eef_ori": identity,
                        },
                        {
                            "observer_seq": 3,
                            "timestamp_s": 3.2,
                            "source_seq": 32,
                            "eef_pos": np.array([2.0, 0.0, 0.0]),
                            "eef_ori": identity,
                        },
                    ]
                )
            with saver.camera_observer._lock:
                saver.camera_observer.buffer.extend(
                    [
                        {
                            "observer_seq": 1,
                            "timestamp_s": 2.0,
                            "camera_frame_seq": 20,
                            "camera_frame_seqs": {"camera_01": 20},
                            "camera_01": np.zeros((2, 2, 3), dtype=np.uint8),
                        },
                        {
                            "observer_seq": 2,
                            "timestamp_s": 2.1,
                            "camera_frame_seq": 21,
                            "camera_frame_seqs": {"camera_01": 21},
                            "camera_01": np.full((2, 2, 3), 10, dtype=np.uint8),
                        },
                        {
                            "observer_seq": 3,
                            "timestamp_s": 2.2,
                            "camera_frame_seq": 22,
                            "camera_frame_seqs": {"camera_01": 22},
                            "camera_01": np.full((2, 2, 3), 20, dtype=np.uint8),
                        },
                    ]
                )

            saver._tick_once()
            saver._tick_once()
            with saver.robot_observer._lock:
                saver.robot_observer.buffer.append(
                    {
                        "observer_seq": 4,
                        "timestamp_s": 3.3,
                        "source_seq": 33,
                        "eef_pos": np.array([3.0, 0.0, 0.0]),
                        "eef_ori": identity,
                    }
                )
            with saver.camera_observer._lock:
                saver.camera_observer.buffer.append(
                    {
                        "observer_seq": 4,
                        "timestamp_s": 2.3,
                        "camera_frame_seq": 23,
                        "camera_frame_seqs": {"camera_01": 23},
                        "camera_01": np.full((2, 2, 3), 30, dtype=np.uint8),
                    }
                )

            saver._tick_once()
            saver._finalize_episode(110.0)
            metadata = saver.last_completed_episode_summary
            self.assertIsNotNone(metadata)
            assert metadata is not None

            lowdim_archive = np.load(saver._episode_dir / "lowdim.npz")
            np.testing.assert_allclose(lowdim_archive["timestamp_s"], np.array([3.2, 3.3]))
            np.testing.assert_allclose(
                lowdim_archive["eef_pos"],
                np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            )
            np.testing.assert_allclose(
                np.load(saver._visual_dir / "camera_01_timestamps.npy"),
                np.array([2.2, 2.3]),
            )
            self.assertEqual(len(writer.frames), 2)
            self.assertEqual(metadata["camera_frame_counts"]["camera_01"], 2)
            self.assertEqual(metadata["camera_duplicate_frame_counts"]["camera_01"], 0)
            self.assertEqual(metadata["lowdim_timestamp_domain"], "sim_time_s")
            self.assertEqual(metadata["camera_timestamp_domains"]["camera_01"], "sim_time_s")
            self.assertEqual(metadata["lowdim_source_time_range_s"], {"start": 3.2, "end": 3.3})
            self.assertEqual(
                metadata["camera_source_time_ranges_s"]["camera_01"],
                {"start": 2.2, "end": 2.3},
            )
    def test_saver_infers_part_metadata_from_scene_xml(self) -> None:
        identity = np.eye(3, dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            models_dir = root / "models"
            asset_dir = root / "generated_cad" / "demo_part"
            models_dir.mkdir(parents=True, exist_ok=False)
            asset_dir.mkdir(parents=True, exist_ok=False)
            (root / "contract.yaml").write_text("robot:\n  xml_path: models/scene.xml\n", encoding="utf-8")
            (models_dir / "scene.xml").write_text(
                "<mujoco model=\"scene\">\n"
                "  <include file=\"fr3.xml\"/>\n"
                "  <include file=\"demo_part.xml\"/>\n"
                "</mujoco>\n",
                encoding="utf-8",
            )
            (models_dir / "fr3.xml").write_text("<mujoco model=\"fr3\"><worldbody/></mujoco>\n", encoding="utf-8")
            (models_dir / "demo_part.xml").write_text(
                "<mujoco model=\"demo_part\">\n"
                "  <asset>\n"
                "    <mesh name=\"demo\" file=\"../generated_cad/demo_part/hole_block.stl\"/>\n"
                "  </asset>\n"
                "  <worldbody>\n"
                "    <body name=\"task\" pos=\"0.4 0.1 0.3\" quat=\"1 0 0 0\"/>\n"
                "  </worldbody>\n"
                "</mujoco>\n",
                encoding="utf-8",
            )
            contract = self._make_contract()
            contract["robot"]["xml_path"] = "models/scene.xml"
            saver, _ = self._prepare_saver(
                tmp_dir=tmp_dir,
                lowdim_buffer=[
                    {
                        "observer_seq": 0,
                        "timestamp_s": 3.0,
                        "source_seq": 30,
                        "eef_pos": np.array([0.0, 0.0, 0.0]),
                        "eef_ori": identity,
                    }
                ],
                camera_buffer=[
                    {
                        "observer_seq": 0,
                        "timestamp_s": 2.0,
                        "camera_frame_seq": 20,
                        "camera_frame_seqs": {"camera_01": 20},
                        "camera_01": np.zeros((2, 2, 3), dtype=np.uint8),
                    }
                ],
                contract=contract,
                contract_path=root / "contract.yaml",
            )

            saver._finalize_episode(110.0)
            metadata = saver.last_completed_episode_summary
            self.assertIsNotNone(metadata)
            assert metadata is not None

            part_metadata = metadata["part_metadata"]
            self.assertEqual(part_metadata["part_name"], "demo_part")
            self.assertEqual(part_metadata["scene_xml_path"], str((models_dir / "scene.xml").resolve()))
            self.assertEqual(part_metadata["part_xml_path"], str((models_dir / "demo_part.xml").resolve()))
            self.assertEqual(part_metadata["part_asset_root"], str(asset_dir.resolve()))
            self.assertEqual(part_metadata["part_position"], [0.4, 0.1, 0.3])
            self.assertEqual(part_metadata["part_orientation_quat"], [1.0, 0.0, 0.0, 0.0])

if __name__ == "__main__":
    unittest.main()
