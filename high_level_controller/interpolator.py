from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp


class InterpolatorFault(RuntimeError):
    pass


@dataclass(frozen=True)
class _PlanSample:
    pos: np.ndarray
    quat: np.ndarray
    force_magnitude: float
    dx_world: np.ndarray


class _Plan:
    def __init__(self, actions: np.ndarray, ts: np.ndarray) -> None:
        self.ts = np.asarray(ts, dtype=float).reshape(-1)
        self.pos = np.asarray(actions[:, :3], dtype=float)
        self.quat = _prepare_quaternions(actions[:, 3:7])
        self.force_magnitudes = np.asarray(actions[:, 7], dtype=float).reshape(-1)
        self.pos_splines = [CubicSpline(self.ts, self.pos[:, axis]) for axis in range(3)]
        self.slerp = Slerp(self.ts, Rotation.from_quat(self.quat))

    def sample(self, now: float) -> _PlanSample:
        if now <= self.ts[0]:
            return _PlanSample(
                pos=self.pos[0].copy(),
                quat=self.quat[0].copy(),
                force_magnitude=float(self.force_magnitudes[0]),
                dx_world=self._segment_dx(0),
            )
        if now >= self.ts[-1]:
            return _PlanSample(
                pos=self.pos[-1].copy(),
                quat=self.quat[-1].copy(),
                force_magnitude=float(self.force_magnitudes[-1]),
                dx_world=self._segment_dx(len(self.ts) - 2),
            )

        segment_idx = self._segment_index(now)
        pos = np.asarray([float(spline(now)) for spline in self.pos_splines], dtype=float)
        quat = self.slerp([float(now)]).as_quat()[0]
        force_magnitude = float(np.interp(now, self.ts, self.force_magnitudes))
        return _PlanSample(
            pos=pos,
            quat=_normalize_quat(quat),
            force_magnitude=force_magnitude,
            dx_world=self._segment_dx(segment_idx),
        )

    def _segment_index(self, now: float) -> int:
        return int(np.clip(np.searchsorted(self.ts, now, side="right") - 1, 0, len(self.ts) - 2))

    def _segment_dx(self, segment_idx: int) -> np.ndarray:
        idx = int(np.clip(segment_idx, 0, len(self.pos) - 2))
        return np.asarray(self.pos[idx + 1] - self.pos[idx], dtype=float)


class TrajectoryInterpolator:
    def __init__(
        self,
        redis_client,
        desired_position_key: str,
        desired_orientation_key: str,
        desired_force_key: str,
        force_dimension_key: str,
        force_or_motion_axis_key: str,
        desired_force_magnitude_key: str | None = None,
        publish_rate_hz: float = 100.0,
        blend_duration: float = 0.1,
    ) -> None:
        if publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive.")
        if blend_duration < 0.0:
            raise ValueError("blend_duration must be non-negative.")

        self.redis_client = redis_client
        self.desired_position_key = desired_position_key
        self.desired_orientation_key = desired_orientation_key
        self.desired_force_key = desired_force_key
        self.force_dimension_key = force_dimension_key
        self.force_or_motion_axis_key = force_or_motion_axis_key
        self.desired_force_magnitude_key = (
            None if desired_force_magnitude_key is None else str(desired_force_magnitude_key)
        )
        self.publish_rate_hz = float(publish_rate_hz)
        self.blend_duration = float(blend_duration)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active: _Plan | None = None
        self._pending: _Plan | None = None
        self._blend: tuple[_Plan, _Plan, float, float] | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._publisher_loop, name="trajectory-interpolator", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def enqueue_chunk(self, A_c, ts) -> None:
        now = time.monotonic()
        actions, ts = self._validate_chunk(A_c, ts, now)
        plan = _Plan(actions, ts)
        with self._lock:
            if self._active is None:
                self._active = plan
                return
            if self._pending is not None:
                raise InterpolatorFault("Received a new chunk while another pending chunk already exists.")
            self._pending = plan

    def _validate_chunk(self, A_c, ts, now: float) -> tuple[np.ndarray, np.ndarray]:
        actions = np.asarray(A_c, dtype=float)
        ts = np.asarray(ts, dtype=float).reshape(-1)
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise InterpolatorFault("A_c must have shape (N, 8).")
        if ts.ndim != 1 or len(ts) != len(actions):
            raise InterpolatorFault("ts must have shape (N,) and match the chunk length.")
        if len(actions) < 2:
            raise InterpolatorFault("Chunks must contain at least 2 waypoints.")
        if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(ts)):
            raise InterpolatorFault("Chunk actions and ts must be finite.")
        if np.any(actions[:, 7] < 0.0):
            raise InterpolatorFault("magnitude_force must be non-negative.")
        if np.any(np.diff(ts) <= 0.0):
            raise InterpolatorFault("ts must be strictly increasing.")
        if ts[-1] <= now:
            raise InterpolatorFault("Received a fully stale chunk.")
        actions = np.array(actions, copy=True)
        actions[:, 3:7] = _prepare_quaternions(actions[:, 3:7])
        return actions, ts

    def _publisher_loop(self) -> None:
        period = 1.0 / self.publish_rate_hz
        while not self._stop_event.is_set():
            start = time.monotonic()
            sample = self._sample(start)
            self._publish_sample(sample)
            sleep_time = period - (time.monotonic() - start)
            if sleep_time > 0.0:
                self._stop_event.wait(sleep_time)

    def _publish_sample(self, sample: _PlanSample | None) -> None:
        if sample is not None:
            _write_vector(self.redis_client, self.desired_position_key, sample.pos)
            _write_matrix(
                self.redis_client,
                self.desired_orientation_key,
                Rotation.from_quat(sample.quat).as_matrix(),
            )
            desired_force = self._desired_force_from_sample(
                sample.force_magnitude,
                sample.dx_world,
            )
            desired_force_magnitude = float(sample.force_magnitude)
        else:
            desired_force = np.zeros(3, dtype=float)
            desired_force_magnitude = 0.0
        _write_vector(self.redis_client, self.desired_force_key, desired_force)
        if self.desired_force_magnitude_key is not None:
            _write_scalar(self.redis_client, self.desired_force_magnitude_key, desired_force_magnitude)

    def _desired_force_from_sample(
        self,
        force_magnitude: float,
        dx_world: np.ndarray,
    ) -> np.ndarray:
        force_dimension = _read_int(self.redis_client, self.force_dimension_key)
        if force_dimension != 1:
            return np.zeros(3, dtype=float)

        force_axis = _normalize_axis(_read_vector(self.redis_client, self.force_or_motion_axis_key))
        projected_component = float(np.dot(np.asarray(dx_world, dtype=float).reshape(3), force_axis))
        if abs(projected_component) <= 1e-12:
            return np.zeros(3, dtype=float)
        desired_force = float(force_magnitude) * np.sign(projected_component) * force_axis
        negative_z_axis = np.array([0.0, 0.0, -1.0], dtype=float)
        if np.linalg.norm(desired_force) > 0.0 and float(np.dot(desired_force, negative_z_axis)) < 0.0:
            desired_force = -desired_force
        return desired_force

    def _sample(self, now: float) -> _PlanSample | None:
        with self._lock:
            active = self._active
            pending = self._pending
            blend = self._blend

            if blend is not None and now >= blend[3]:
                self._active = blend[1]
                self._blend = None
                active = self._active
                blend = None

            if blend is None and pending is not None and now >= pending.ts[0]:
                if now >= pending.ts[-1]:
                    self._pending = None
                    raise InterpolatorFault("Pending chunk became stale before it could be blended.")
                if active is None:
                    self._active = pending
                    self._pending = None
                    active = self._active
                elif self.blend_duration <= 0.0:
                    self._active = pending
                    self._pending = None
                    active = self._active
                else:
                    self._blend = (active, pending, now, now + self.blend_duration)
                    self._pending = None
                    blend = self._blend

            if blend is None and active is not None and pending is None and now > active.ts[-1]:
                self._active = None
                active = None

        if blend is not None:
            old_plan, new_plan, start, end = blend
            old_sample = old_plan.sample(now)
            new_sample = new_plan.sample(now)
            alpha = _min_jerk_alpha(now, start, end)
            pos = (1.0 - alpha) * old_sample.pos + alpha * new_sample.pos
            quat = _blend_quaternions(old_sample.quat, new_sample.quat, alpha)
            force_magnitude = (1.0 - alpha) * old_sample.force_magnitude + alpha * new_sample.force_magnitude
            dx_world = (1.0 - alpha) * old_sample.dx_world + alpha * new_sample.dx_world
            return _PlanSample(
                pos=pos,
                quat=quat,
                force_magnitude=float(force_magnitude),
                dx_world=np.asarray(dx_world, dtype=float),
            )

        if active is None:
            return None
        return active.sample(now)


def _redis_text(value: bytes | str | None) -> str:
    if value is None:
        raise InterpolatorFault("Requested Redis key is missing.")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_json_value(redis_client, key: str):
    return json.loads(_redis_text(redis_client.get(key)))


def _read_vector(redis_client, key: str) -> np.ndarray:
    vector = np.asarray(_read_json_value(redis_client, key), dtype=float).reshape(-1)
    if vector.size != 3:
        raise InterpolatorFault(f"Redis key `{key}` did not contain a 3D vector.")
    return vector.astype(float)


def _read_int(redis_client, key: str) -> int:
    raw_value = redis_client.get(key)
    if raw_value is None:
        raise InterpolatorFault(f"Requested Redis key `{key}` is missing.")
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8")
    return int(raw_value)


def _write_vector(redis_client, key: str, value: np.ndarray) -> None:
    redis_client.set(key, json.dumps(np.asarray(value, dtype=float).reshape(3).tolist()))


def _write_matrix(redis_client, key: str, value: np.ndarray) -> None:
    matrix = np.asarray(value, dtype=float).reshape(3, 3)
    redis_client.set(key, json.dumps(matrix.tolist()))


def _write_scalar(redis_client, key: str, value: float) -> None:
    redis_client.set(key, json.dumps([float(value)]))


def _normalize_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return axis / norm


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).reshape(4)
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        raise InterpolatorFault("Quaternion norm must be positive.")
    return quat / norm


def _prepare_quaternions(quats) -> np.ndarray:
    quats = np.asarray(quats, dtype=float)
    prepared = np.zeros_like(quats, dtype=float)
    prepared[0] = _normalize_quat(quats[0])
    for idx in range(1, len(quats)):
        quat = _normalize_quat(quats[idx])
        if np.dot(prepared[idx - 1], quat) < 0.0:
            quat = -quat
        prepared[idx] = quat
    return prepared


def _blend_quaternions(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    slerp = Slerp([0.0, 1.0], Rotation.from_quat([q0, q1]))
    return _normalize_quat(slerp([float(alpha)]).as_quat()[0])


def _min_jerk_alpha(now: float, start: float, end: float) -> float:
    duration = max(end - start, 1e-6)
    s = float(np.clip((now - start) / duration, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
