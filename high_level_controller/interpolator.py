from __future__ import annotations

import json
import threading
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp


class InterpolatorFault(RuntimeError):
    pass


class _Plan:
    def __init__(self, actions: np.ndarray, ts: np.ndarray) -> None:
        self.ts = np.asarray(ts, dtype=float).reshape(-1)
        self.pos = np.asarray(actions[:, :3], dtype=float)
        self.quat = _prepare_quaternions(actions[:, 3:7])
        self.pos_splines = [CubicSpline(self.ts, self.pos[:, axis]) for axis in range(3)]
        self.slerp = Slerp(self.ts, Rotation.from_quat(self.quat))

    def sample(self, now: float) -> tuple[np.ndarray, np.ndarray]:
        if now <= self.ts[0]:
            return self.pos[0].copy(), self.quat[0].copy()
        if now >= self.ts[-1]:
            return self.pos[-1].copy(), self.quat[-1].copy()
        pos = np.asarray([float(spline(now)) for spline in self.pos_splines], dtype=float)
        quat = self.slerp([float(now)]).as_quat()[0]
        return pos, _normalize_quat(quat)


class TrajectoryInterpolator:
    def __init__(
        self,
        redis_client,
        desired_position_key: str,
        desired_orientation_key: str,
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
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise InterpolatorFault("A_c must have shape (N, 7).")
        if ts.ndim != 1 or len(ts) != len(actions):
            raise InterpolatorFault("ts must have shape (N,) and match the chunk length.")
        if len(actions) < 2:
            raise InterpolatorFault("Chunks must contain at least 2 waypoints.")
        if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(ts)):
            raise InterpolatorFault("Chunk actions and ts must be finite.")
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
            if sample is not None:
                pos, quat = sample
                self.redis_client.set(self.desired_position_key, json.dumps(pos.tolist()))
                rot = Rotation.from_quat(quat).as_matrix().tolist()
                self.redis_client.set(self.desired_orientation_key, json.dumps(rot))
            sleep_time = period - (time.monotonic() - start)
            if sleep_time > 0.0:
                self._stop_event.wait(sleep_time)

    def _sample(self, now: float) -> tuple[np.ndarray, np.ndarray] | None:
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
            old_pos, old_quat = old_plan.sample(now)
            new_pos, new_quat = new_plan.sample(now)
            alpha = _min_jerk_alpha(now, start, end)
            pos = (1.0 - alpha) * old_pos + alpha * new_pos
            quat = _blend_quaternions(old_quat, new_quat, alpha)
            return pos, quat

        if active is None:
            return None
        return active.sample(now)


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
