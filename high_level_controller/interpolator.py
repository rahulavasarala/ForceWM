from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp


_TIME_EPS = 1e-6


class InterpolatorFault(RuntimeError):
    """Raised when the interpolator enters a faulted state."""


@dataclass
class PoseSample:
    t_abs: float
    position: NDArray[np.float64]
    quaternion: NDArray[np.float64]
    linear_velocity: NDArray[np.float64]
    linear_acceleration: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.position = _as_vector(self.position, "position")
        self.quaternion = _normalize_quaternion(
            _as_vector(self.quaternion, "quaternion", expected_size=4)
        )
        self.linear_velocity = _as_vector(
            self.linear_velocity, "linear_velocity"
        )
        self.linear_acceleration = _as_vector(
            self.linear_acceleration, "linear_acceleration"
        )
        self.t_abs = float(self.t_abs)

    def rotation_matrix(self) -> NDArray[np.float64]:
        return Rotation.from_quat(self.quaternion).as_matrix()

    def rpy(self) -> NDArray[np.float64]:
        return Rotation.from_quat(self.quaternion).as_euler("xyz")


@dataclass
class _QueuedChunk:
    waypoints: NDArray[np.float64]
    first_waypoint_time: float


@dataclass
class _Transition:
    old_plan: "_ClampedPlan"
    new_plan: "_ClampedPlan"
    start_time: float
    end_time: float


@dataclass
class _ClampedPlan:
    knot_times: NDArray[np.float64]
    positions: NDArray[np.float64]
    quaternions: NDArray[np.float64]
    start_velocity: NDArray[np.float64]
    end_velocity: NDArray[np.float64]
    _position_splines: tuple[CubicSpline, CubicSpline, CubicSpline] = field(
        init=False, repr=False
    )
    _orientation_slerp: Slerp = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.knot_times = np.asarray(self.knot_times, dtype=float).reshape(-1)
        self.positions = np.asarray(self.positions, dtype=float)
        self.quaternions = _prepare_quaternion_path(self.quaternions)
        self.start_velocity = _as_vector(self.start_velocity, "start_velocity")
        self.end_velocity = _as_vector(self.end_velocity, "end_velocity")

        if self.knot_times.ndim != 1 or self.knot_times.size < 2:
            raise ValueError("Clamped plan requires at least two knot times.")
        if self.positions.shape != (self.knot_times.size, 3):
            raise ValueError("Plan positions must have shape (N, 3).")
        if self.quaternions.shape != (self.knot_times.size, 4):
            raise ValueError("Plan quaternions must have shape (N, 4).")
        if np.any(np.diff(self.knot_times) <= 0.0):
            raise ValueError("Plan knot times must be strictly increasing.")

        self._position_splines = tuple(
            CubicSpline(
                self.knot_times,
                self.positions[:, axis],
                bc_type=(
                    (1, float(self.start_velocity[axis])),
                    (1, float(self.end_velocity[axis])),
                ),
            )
            for axis in range(3)
        )
        self._orientation_slerp = Slerp(
            self.knot_times,
            Rotation.from_quat(self.quaternions),
        )

    @property
    def start_time(self) -> float:
        return float(self.knot_times[0])

    @property
    def end_time(self) -> float:
        return float(self.knot_times[-1])

    def sample(self, now: float) -> PoseSample:
        now = float(now)

        if now <= self.start_time + _TIME_EPS:
            return PoseSample(
                t_abs=now,
                position=self.positions[0].copy(),
                quaternion=self.quaternions[0].copy(),
                linear_velocity=self._derivative(self.start_time, order=1),
                linear_acceleration=self._derivative(self.start_time, order=2),
            )

        if now >= self.end_time - _TIME_EPS:
            return PoseSample(
                t_abs=now,
                position=self.positions[-1].copy(),
                quaternion=self.quaternions[-1].copy(),
                linear_velocity=np.zeros(3),
                linear_acceleration=np.zeros(3),
            )

        return PoseSample(
            t_abs=now,
            position=self._position(now),
            quaternion=self._orientation(now),
            linear_velocity=self._derivative(now, order=1),
            linear_acceleration=self._derivative(now, order=2),
        )

    def _position(self, now: float) -> NDArray[np.float64]:
        return np.asarray(
            [float(spline(now)) for spline in self._position_splines],
            dtype=float,
        )

    def _derivative(self, now: float, order: int) -> NDArray[np.float64]:
        return np.asarray(
            [float(spline(now, order)) for spline in self._position_splines],
            dtype=float,
        )

    def _orientation(self, now: float) -> NDArray[np.float64]:
        return _normalize_quaternion(
            self._orientation_slerp([float(now)]).as_quat()[0]
        )


class TrajectoryInterpolator:
    def __init__(
        self,
        command_sink: Callable[[PoseSample], None],
        state_source: Optional[Callable[[], PoseSample]] = None,
        send_rate_hz: float = 100.0,
        blend_duration: float = 0.0,
        max_chunks: int = 2,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if send_rate_hz <= 0.0:
            raise ValueError("send_rate_hz must be positive.")
        if blend_duration < 0.0:
            raise ValueError("blend_duration must be non-negative.")
        if max_chunks not in (1, 2):
            raise ValueError("max_chunks must be 1 or 2 for this interpolator.")

        self._command_sink = command_sink
        self._state_source = state_source
        self._send_rate_hz = float(send_rate_hz)
        self._blend_duration = float(blend_duration)
        self._max_chunks = int(max_chunks)
        self._clock = clock
        self._sleep_fn = sleep_fn

        self._active_plan: Optional[_ClampedPlan] = None
        self._pending_chunk: Optional[_QueuedChunk] = None
        self._transition: Optional[_Transition] = None
        self._initial_state: Optional[PoseSample] = None
        self._last_commanded_sample: Optional[PoseSample] = None
        self._stop_requested = False
        self._faulted = False
        self._fault_reason: Optional[str] = None

    def set_initial_state(self, sample: PoseSample) -> None:
        self._initial_state = PoseSample(
            t_abs=sample.t_abs,
            position=sample.position.copy(),
            quaternion=sample.quaternion.copy(),
            linear_velocity=sample.linear_velocity.copy(),
            linear_acceleration=sample.linear_acceleration.copy(),
        )

    def enqueue_chunk(self, chunk: ArrayLike) -> None:
        now = float(self._clock())
        self._promote_pending_if_ready(now)
        chunk_array = self._validate_chunk(chunk)

        if self._active_plan is None:
            start_sample = self._bootstrap_start_sample(now)
            prepared_chunk = self._prepare_chunk_for_build(chunk_array, now)
            self._active_plan = self._build_plan(
                prepared_chunk,
                start_sample,
            )
            return

        occupied_slots = 1 + int(self._pending_chunk is not None)
        if occupied_slots >= self._max_chunks:
            self._fault(
                "Received more action chunks than the interpolator queue allows."
            )

        self._pending_chunk = _QueuedChunk(
            waypoints=np.array(chunk_array, copy=True),
            first_waypoint_time=float(chunk_array[0, 6]),
        )

    def sample(self, now: float) -> PoseSample:
        now = float(now)
        self._promote_pending_if_ready(now)

        if self._transition is not None:
            if now < self._transition.end_time - _TIME_EPS:
                return self._sample_transition(now)
            self._transition = None

        if self._active_plan is not None:
            return self._active_plan.sample(now)

        return self._hold_sample(now)

    def run(self) -> None:
        self._stop_requested = False
        period = 1.0 / self._send_rate_hz

        while not self._stop_requested and not self._faulted:
            loop_start = float(self._clock())
            sample = self.sample(loop_start)
            self._command_sink(sample)
            self._last_commanded_sample = sample

            elapsed = float(self._clock()) - loop_start
            remaining = period - elapsed
            if remaining > 0.0:
                self._sleep_fn(remaining)

    def stop(self) -> None:
        self._stop_requested = True

    def is_faulted(self) -> bool:
        return self._faulted

    def fault_reason(self) -> Optional[str]:
        return self._fault_reason

    def _validate_chunk(self, chunk: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(chunk, dtype=float)
        if array.ndim != 2 or array.shape[1] != 7:
            self._fault("Action chunk must be an N x 7 array-like object.")
        if array.shape[0] < 1:
            self._fault("Action chunk must contain at least one waypoint.")
        if not np.all(np.isfinite(array)):
            self._fault("Action chunk contains non-finite values.")
        if np.any(np.diff(array[:, 6]) <= 0.0):
            self._fault("Waypoint times must be strictly increasing.")
        return array

    def _prepare_chunk_for_build(
        self,
        chunk: NDArray[np.float64],
        start_time: float,
    ) -> NDArray[np.float64]:
        future_mask = chunk[:, 6] >= start_time
        if not np.any(future_mask):
            self._fault("Action chunk is fully stale and cannot be enqueued.")

        prepared = np.array(chunk[future_mask], copy=True)
        minimum_time = float(start_time) + _TIME_EPS
        for idx in range(prepared.shape[0]):
            if prepared[idx, 6] < minimum_time:
                prepared[idx, 6] = minimum_time
            minimum_time = prepared[idx, 6] + _TIME_EPS

        return prepared

    def _bootstrap_start_sample(self, now: float) -> PoseSample:
        if self._initial_state is None:
            if self._state_source is None:
                self._fault(
                    "Interpolator needs an initial state or state_source before the first chunk."
                )

            source_sample = self._state_source()
            self._initial_state = PoseSample(
                t_abs=now,
                position=source_sample.position.copy(),
                quaternion=source_sample.quaternion.copy(),
                linear_velocity=np.zeros(3),
                linear_acceleration=np.zeros(3),
            )

        return PoseSample(
            t_abs=now,
            position=self._initial_state.position.copy(),
            quaternion=self._initial_state.quaternion.copy(),
            linear_velocity=np.zeros(3),
            linear_acceleration=np.zeros(3),
        )

    def _build_plan(
        self,
        chunk: NDArray[np.float64],
        start_sample: PoseSample,
    ) -> _ClampedPlan:
        waypoint_times = np.asarray(chunk[:, 6], dtype=float)
        positions = np.vstack((start_sample.position, chunk[:, :3]))
        quaternions = np.vstack(
            (
                start_sample.quaternion,
                Rotation.from_euler("xyz", chunk[:, 3:6]).as_quat(),
            )
        )
        knot_times = np.concatenate(([start_sample.t_abs], waypoint_times))

        return _ClampedPlan(
            knot_times=knot_times,
            positions=positions,
            quaternions=quaternions,
            start_velocity=start_sample.linear_velocity,
            end_velocity=np.zeros(3),
        )

    def _promote_pending_if_ready(self, now: float) -> None:
        if (
            self._pending_chunk is None
            or now < self._pending_chunk.first_waypoint_time - _TIME_EPS
        ):
            return

        start_sample = (
            self._active_plan.sample(now)
            if self._active_plan is not None
            else self._hold_sample(now)
        )
        prepared_chunk = self._prepare_chunk_for_build(
            self._pending_chunk.waypoints,
            now,
        )
        new_plan = self._build_plan(prepared_chunk, start_sample)
        old_plan = self._active_plan

        self._active_plan = new_plan
        self._pending_chunk = None

        if old_plan is None or self._blend_duration <= 0.0:
            self._transition = None
            return

        self._transition = _Transition(
            old_plan=old_plan,
            new_plan=new_plan,
            start_time=now,
            end_time=now + self._blend_duration,
        )

    def _sample_transition(self, now: float) -> PoseSample:
        assert self._transition is not None

        old_sample = self._transition.old_plan.sample(now)
        new_sample = self._transition.new_plan.sample(now)
        duration = self._transition.end_time - self._transition.start_time
        s = np.clip((now - self._transition.start_time) / duration, 0.0, 1.0)
        alpha, alpha_dot, alpha_ddot = _min_jerk_blend_weight(s, duration)

        delta_position = new_sample.position - old_sample.position
        delta_velocity = new_sample.linear_velocity - old_sample.linear_velocity

        position = (
            (1.0 - alpha) * old_sample.position + alpha * new_sample.position
        )
        linear_velocity = (
            (1.0 - alpha) * old_sample.linear_velocity
            + alpha * new_sample.linear_velocity
            + alpha_dot * delta_position
        )
        linear_acceleration = (
            (1.0 - alpha) * old_sample.linear_acceleration
            + alpha * new_sample.linear_acceleration
            + 2.0 * alpha_dot * delta_velocity
            + alpha_ddot * delta_position
        )
        quaternion = _quat_slerp(old_sample.quaternion, new_sample.quaternion, alpha)

        return PoseSample(
            t_abs=now,
            position=position,
            quaternion=quaternion,
            linear_velocity=linear_velocity,
            linear_acceleration=linear_acceleration,
        )

    def _hold_sample(self, now: float) -> PoseSample:
        if self._last_commanded_sample is not None:
            return PoseSample(
                t_abs=now,
                position=self._last_commanded_sample.position.copy(),
                quaternion=self._last_commanded_sample.quaternion.copy(),
                linear_velocity=np.zeros(3),
                linear_acceleration=np.zeros(3),
            )

        if self._initial_state is not None:
            return PoseSample(
                t_abs=now,
                position=self._initial_state.position.copy(),
                quaternion=self._initial_state.quaternion.copy(),
                linear_velocity=np.zeros(3),
                linear_acceleration=np.zeros(3),
            )

        if self._state_source is not None:
            source_sample = self._state_source()
            self._initial_state = PoseSample(
                t_abs=now,
                position=source_sample.position.copy(),
                quaternion=source_sample.quaternion.copy(),
                linear_velocity=np.zeros(3),
                linear_acceleration=np.zeros(3),
            )
            return self._hold_sample(now)

        self._fault("Interpolator has no active chunk and no state to hold.")
        raise AssertionError("unreachable")

    def _fault(self, reason: str) -> None:
        self._faulted = True
        self._fault_reason = reason
        self._stop_requested = True
        raise InterpolatorFault(reason)


def make_redis_command_sink(
    redis_client,
    desired_position_key: str,
    desired_orientation_key: str,
) -> Callable[[PoseSample], None]:
    def sink(sample: PoseSample) -> None:
        redis_client.set(
            desired_position_key,
            json.dumps(sample.position.tolist()),
        )
        redis_client.set(
            desired_orientation_key,
            json.dumps(sample.rotation_matrix().tolist()),
        )

    return sink


def make_redis_state_source(
    redis_client,
    current_position_key: str,
    current_orientation_key: str,
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[], PoseSample]:
    def source() -> PoseSample:
        position = np.asarray(
            json.loads(_redis_text(redis_client.get(current_position_key))),
            dtype=float,
        ).reshape(3)
        rotation_matrix = np.asarray(
            json.loads(_redis_text(redis_client.get(current_orientation_key))),
            dtype=float,
        ).reshape(3, 3)

        return PoseSample(
            t_abs=float(clock()),
            position=position,
            quaternion=Rotation.from_matrix(rotation_matrix).as_quat(),
            linear_velocity=np.zeros(3),
            linear_acceleration=np.zeros(3),
        )

    return source


def _as_vector(
    value: ArrayLike,
    name: str,
    expected_size: int = 3,
) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != expected_size:
        raise ValueError(f"{name} must have exactly {expected_size} elements.")
    return vector


def _normalize_quaternion(quaternion: ArrayLike) -> NDArray[np.float64]:
    quat = np.asarray(quaternion, dtype=float).reshape(4)
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive.")
    return quat / norm


def _prepare_quaternion_path(
    quaternions: ArrayLike,
) -> NDArray[np.float64]:
    quats = np.asarray(quaternions, dtype=float).reshape(-1, 4).copy()
    quats[0] = _normalize_quaternion(quats[0])

    for idx in range(1, quats.shape[0]):
        quats[idx] = _normalize_quaternion(quats[idx])
        if np.dot(quats[idx - 1], quats[idx]) < 0.0:
            quats[idx] = -quats[idx]

    return quats


def _quat_slerp(
    q0: ArrayLike,
    q1: ArrayLike,
    alpha: float,
) -> NDArray[np.float64]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    quat0 = _normalize_quaternion(q0)
    quat1 = _normalize_quaternion(q1)
    dot = float(np.dot(quat0, quat1))

    if dot < 0.0:
        quat1 = -quat1
        dot = -dot

    if dot > 0.9995:
        return _normalize_quaternion((1.0 - alpha) * quat0 + alpha * quat1)

    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * alpha
    sin_theta = float(np.sin(theta))

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _normalize_quaternion(s0 * quat0 + s1 * quat1)


def _min_jerk_blend_weight(
    s: float,
    duration: float,
) -> tuple[float, float, float]:
    s = float(np.clip(s, 0.0, 1.0))

    alpha = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    dalpha_ds = 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4
    d2alpha_ds2 = 60.0 * s - 180.0 * s**2 + 120.0 * s**3

    alpha_dot = dalpha_ds / duration
    alpha_ddot = d2alpha_ds2 / (duration**2)
    return alpha, alpha_dot, alpha_ddot


def _redis_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
