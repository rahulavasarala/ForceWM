from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import redis
import yaml


def _default_contract_path() -> Path:
    return Path(__file__).resolve().parents[1] / "universal_contract.yaml"


class RobotObserver:
    def __init__(self, buffer_size, example_obs, obs_freq, robot_data):
        # creates a ring buffer of size buffer_size to store the last buffer_size observations
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")

        self.buffer_size = int(buffer_size)
        self.example_obs = example_obs
        self.contract = self._load_contract(robot_data)
        self.lowdim_specs = self._parse_lowdim_specs(self.contract)

        if len(self.lowdim_specs) == 0:
            raise ValueError("No lowdim keys were found in the contract.")

        self.obs_freq = float(obs_freq) if obs_freq is not None else self._resolve_default_obs_freq()
        if self.obs_freq <= 0.0:
            raise ValueError("obs_freq must be positive.")

        self.obs_period_s = 1.0 / self.obs_freq
        self.buffer = deque(maxlen=self.buffer_size)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.redis_client = redis.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=False)

    def get_last_k_obs(self, k):
        # create a dictionary that batches the observations in the ring buffer into a single dictionary
        # keyed by lowdim name, with time stacked along axis 0 when possible
        if k <= 0:
            raise ValueError("k must be positive.")

        with self._lock:
            recent_observations = list(self.buffer)[-k:]

        if not recent_observations:
            return {}

        batched: dict[str, Any] = {}
        batched["timestamp_s"] = np.asarray(
            [obs["timestamp_s"] for obs in recent_observations], dtype=np.float64
        )
        for lowdim_name in self.lowdim_specs:
            values = [obs[lowdim_name] for obs in recent_observations]
            batched[lowdim_name] = self._stack_or_list(values)

        return batched

    def start_adding_obs(self):
        # starts a thread that continuously adds observations at obs_freq to the ring buffer until stop_adding_obs is called
        if self._thread is not None and self._thread.is_alive():
            return

        self.redis_client.ping()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._observation_loop, name="robot-observer", daemon=True)
        self._thread.start()

    def stop_adding_obs(self):
        # stops the thread that is adding observations to the ring buffer
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _observation_loop(self) -> None:
        next_poll_time = time.perf_counter()

        while not self._stop_event.is_set():
            observation = self._read_observation_from_redis()
            if observation is not None:
                observation["timestamp_s"] = time.time()
                with self._lock:
                    self.buffer.append(observation)

            next_poll_time += self.obs_period_s
            sleep_duration = next_poll_time - time.perf_counter()
            if sleep_duration > 0.0:
                self._stop_event.wait(sleep_duration)
            else:
                next_poll_time = time.perf_counter()

    def _read_observation_from_redis(self) -> dict[str, Any] | None:
        observation: dict[str, Any] = {}
        for lowdim_name, lowdim_spec in self.lowdim_specs.items():
            redis_value = self.redis_client.get(lowdim_spec["redis_key"])
            if redis_value is None:
                return None

            observation[lowdim_name] = np.array(
                json.loads(redis_value)
            )

        return observation

    def _resolve_default_obs_freq(self) -> float:
        robot_cfg = self.contract.get("robot", {})
        lowdim_cfg = robot_cfg.get("data_sources", {}).get("lowdim", {})
        fps = lowdim_cfg.get("fps")

        if fps is None:
            raise ValueError("obs_freq was not provided and no lowdim fps was found in the contract.")
        return float(fps)

    @staticmethod
    def _load_contract(robot_data: Any) -> dict[str, Any]:
        if isinstance(robot_data, dict):
            if "contract" in robot_data:
                robot_data = robot_data["contract"]
            if isinstance(robot_data, dict) and "robot" in robot_data:
                return robot_data
            if isinstance(robot_data, dict):
                return {"robot": robot_data}

        if robot_data is None:
            contract_path = _default_contract_path()
        else:
            contract_path = Path(robot_data)

        with contract_path.open("r", encoding="utf-8") as handle:
            contract = yaml.safe_load(handle)

        if not isinstance(contract, dict):
            raise ValueError("The universal contract must load as a dictionary.")

        return contract

    @staticmethod
    def _parse_lowdim_specs(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
        robot_cfg = contract.get("robot")
        if not isinstance(robot_cfg, dict):
            raise ValueError("The universal contract must contain a top-level `robot` mapping.")

        prefix = robot_cfg.get("prefix")
        if prefix is None:
            raise ValueError("The robot is missing a prefix in the universal contract.")
        redis_namespace = RobotObserver._normalize_redis_namespace(
            robot_cfg.get("redis_namespace", "sai")
        )

        lowdim_cfg = robot_cfg.get("data_sources", {}).get("lowdim", {})
        lowdim_keys = lowdim_cfg.get("keys", [])

        parsed_lowdim: dict[str, dict[str, Any]] = {}
        for lowdim_entry in lowdim_keys:
            if not isinstance(lowdim_entry, dict) or len(lowdim_entry) != 1:
                continue

            lowdim_name, lowdim_spec = next(iter(lowdim_entry.items()))
            redis_suffix = lowdim_spec.get("redis", lowdim_name)
            parsed_lowdim[lowdim_name] = {
                "redis_key": RobotObserver._make_redis_key(
                    redis_namespace, prefix, redis_suffix
                ),
                "dim": lowdim_spec.get("dim"),
            }

        return parsed_lowdim

    @staticmethod
    def _normalize_redis_namespace(redis_namespace: Any) -> str:
        if redis_namespace is None:
            return ""
        return str(redis_namespace).strip(":")

    @staticmethod
    def _make_redis_key(redis_namespace: str, prefix: str, suffix: str) -> str:
        redis_key = f"{prefix.rstrip(':')}::{suffix.lstrip(':')}"
        if not redis_namespace:
            return redis_key
        return f"{redis_namespace}::{redis_key}"

    @staticmethod
    def _stack_or_list(values: list[Any]) -> Any:
        try:
            arrays = [np.asarray(value) for value in values]
            return np.stack(arrays, axis=0)
        except Exception:
            return values
