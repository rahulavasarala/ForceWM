from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import redis

try:
    from .interpolator import (
        InterpolatorFault,
        PoseSample,
        TrajectoryInterpolator,
        make_redis_command_sink,
    )
except ImportError:
    from interpolator import (  # type: ignore
        InterpolatorFault,
        PoseSample,
        TrajectoryInterpolator,
        make_redis_command_sink,
    )


DEFAULT_POSITION_KEY = "test::interpolator::desired_cartesian_position"
DEFAULT_ORIENTATION_KEY = "test::interpolator::desired_cartesian_orientation"


@dataclass
class TestConfig:
    num_chunks: int
    action_frequency_hz: float
    interpolator_frequency_hz: float
    overlap_seconds: float
    points_per_chunk: int
    chunk_duration: float
    first_waypoint_lead: float
    blend_duration: float
    poll_frequency_hz: float
    seed: int
    redis_host: str
    redis_port: int
    redis_db: int
    tail_duration: float
    save_path: Optional[str]

    @property
    def waypoint_dt(self) -> float:
        return self.chunk_duration / float(self.points_per_chunk - 1)


def _parse_args() -> TestConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Visual end-to-end test for the trajectory interpolator. "
            "Random action chunks are sent through the Redis sink, then "
            "the desired position key is read back and plotted."
        )
    )
    parser.add_argument("--num-chunks", type=int, default=6)
    parser.add_argument("--action-frequency-hz", type=float, default=1.0)
    parser.add_argument("--interpolator-frequency-hz", type=float, default=100.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.3)
    parser.add_argument("--points-per-chunk", type=int, default=5)
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=1.4,
        help=(
            "Time from the first waypoint in a chunk to the last waypoint. "
            "The per-waypoint spacing is derived from this and points-per-chunk."
        ),
    )
    parser.add_argument("--first-waypoint-lead", type=float, default=0.0)
    parser.add_argument("--blend-duration", type=float, default=0.0)
    parser.add_argument("--poll-frequency-hz", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--redis-host", type=str, default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--tail-duration", type=float, default=0.75)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    if args.num_chunks < 1:
        raise ValueError("num_chunks must be at least 1.")
    if args.action_frequency_hz <= 0.0:
        raise ValueError("action_frequency_hz must be positive.")
    if args.interpolator_frequency_hz <= 0.0:
        raise ValueError("interpolator_frequency_hz must be positive.")
    if args.overlap_seconds < 0.0:
        raise ValueError("overlap_seconds must be non-negative.")
    if args.points_per_chunk < 2:
        raise ValueError("points_per_chunk must be at least 2.")
    if args.chunk_duration <= 0.0:
        raise ValueError("chunk_duration must be positive.")
    if args.first_waypoint_lead < 0.0:
        raise ValueError("first_waypoint_lead must be non-negative.")
    if args.blend_duration < 0.0:
        raise ValueError("blend_duration must be non-negative.")
    if args.poll_frequency_hz <= 0.0:
        raise ValueError("poll_frequency_hz must be positive.")
    if args.tail_duration < 0.0:
        raise ValueError("tail_duration must be non-negative.")

    return TestConfig(
        num_chunks=args.num_chunks,
        action_frequency_hz=args.action_frequency_hz,
        interpolator_frequency_hz=args.interpolator_frequency_hz,
        overlap_seconds=args.overlap_seconds,
        points_per_chunk=args.points_per_chunk,
        chunk_duration=args.chunk_duration,
        first_waypoint_lead=args.first_waypoint_lead,
        blend_duration=args.blend_duration,
        poll_frequency_hz=args.poll_frequency_hz,
        seed=args.seed,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        tail_duration=args.tail_duration,
        save_path=args.save_path,
    )


def _random_chunk(
    now: float,
    previous_end_time: Optional[float],
    previous_end_position: np.ndarray,
    previous_end_rpy: np.ndarray,
    rng: np.random.Generator,
    config: TestConfig,
) -> np.ndarray:
    waypoint_dt = config.waypoint_dt

    if previous_end_time is None:
        first_waypoint_time = now + config.first_waypoint_lead
    else:
        first_waypoint_time = max(
            now + config.first_waypoint_lead,
            previous_end_time - config.overlap_seconds,
        )

    waypoint_times = first_waypoint_time + waypoint_dt * np.arange(
        config.points_per_chunk,
        dtype=float,
    )

    target_delta = rng.uniform(
        low=np.array([-0.10, -0.10, -0.05]),
        high=np.array([0.10, 0.10, 0.05]),
    )
    target_position = previous_end_position + target_delta
    target_position[2] = np.clip(target_position[2], 0.15, 0.65)

    alphas = np.linspace(0.0, 1.0, config.points_per_chunk)[:, None]
    position_noise = rng.normal(
        loc=0.0,
        scale=np.array([0.008, 0.008, 0.004]),
        size=(config.points_per_chunk, 3),
    )
    positions = (
        (1.0 - alphas) * previous_end_position[None, :]
        + alphas * target_position[None, :]
        + position_noise
    )
    positions[0] = previous_end_position
    positions[:, 2] = np.clip(positions[:, 2], 0.15, 0.65)

    target_rpy = previous_end_rpy + rng.uniform(
        low=np.array([-0.15, -0.15, -0.35]),
        high=np.array([0.15, 0.15, 0.35]),
    )
    rpy_noise = rng.normal(
        loc=0.0,
        scale=np.array([0.01, 0.01, 0.02]),
        size=(config.points_per_chunk, 3),
    )
    rpy = (
        (1.0 - alphas) * previous_end_rpy[None, :]
        + alphas * target_rpy[None, :]
        + rpy_noise
    )
    rpy[0] = previous_end_rpy

    return np.column_stack((positions, rpy, waypoint_times))


def _read_position_key(
    redis_client: redis.Redis,
    position_key: str,
) -> Optional[np.ndarray]:
    raw_value = redis_client.get(position_key)
    if raw_value is None:
        return None
    return np.asarray(json.loads(raw_value), dtype=float).reshape(3)


def _plot_results(
    config: TestConfig,
    action_waypoints: list[np.ndarray],
    logged_positions: np.ndarray,
) -> None:
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")

    for idx, waypoints in enumerate(action_waypoints):
        label = "Action chunk waypoints" if idx == 0 else None
        axis.scatter(
            waypoints[:, 0],
            waypoints[:, 1],
            waypoints[:, 2],
            color="cyan",
            s=35,
            label=label,
        )

    if logged_positions.size > 0:
        axis.plot(
            logged_positions[:, 0],
            logged_positions[:, 1],
            logged_positions[:, 2],
            color="green",
            linewidth=2.0,
            label="Logged Redis trajectory",
        )
        axis.scatter(
            logged_positions[0, 0],
            logged_positions[0, 1],
            logged_positions[0, 2],
            color="green",
            s=45,
        )

    axis.set_title(
        "Interpolator End-to-End Visual Test\n"
        f"chunks={config.num_chunks}, action_hz={config.action_frequency_hz}, "
        f"interp_hz={config.interpolator_frequency_hz}, "
        f"chunk_dur={config.chunk_duration:.2f}s, "
        f"overlap={config.overlap_seconds:.2f}s"
    )
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")
    axis.legend(loc="upper left")
    axis.grid(True)
    axis.set_box_aspect((1.0, 1.0, 0.7))
    figure.tight_layout()

    if config.save_path is not None:
        figure.savefig(config.save_path, dpi=180)

    plt.show()


def main() -> None:
    config = _parse_args()
    rng = np.random.default_rng(config.seed)

    redis_client = redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        db=config.redis_db,
        decode_responses=True,
    )
    redis_client.ping()
    redis_client.delete(DEFAULT_POSITION_KEY, DEFAULT_ORIENTATION_KEY)

    sink = make_redis_command_sink(
        redis_client,
        DEFAULT_POSITION_KEY,
        DEFAULT_ORIENTATION_KEY,
    )
    interpolator = TrajectoryInterpolator(
        command_sink=sink,
        send_rate_hz=config.interpolator_frequency_hz,
        blend_duration=config.blend_duration,
    )

    initial_position = np.array([0.45, 0.0, 0.35], dtype=float)
    initial_rpy = np.zeros(3, dtype=float)
    interpolator.set_initial_state(
        PoseSample(
            t_abs=time.monotonic(),
            position=initial_position,
            quaternion=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            linear_velocity=np.zeros(3),
            linear_acceleration=np.zeros(3),
        )
    )

    action_waypoints: list[np.ndarray] = []
    logged_positions: list[np.ndarray] = []
    thread_error: list[BaseException] = []
    publish_error: list[BaseException] = []
    publish_done = threading.Event()
    end_time_holder: dict[str, Optional[float]] = {"final_end_time": None}

    def run_interpolator() -> None:
        try:
            interpolator.run()
        except BaseException as exc:
            thread_error.append(exc)

    def publish_chunks() -> None:
        previous_end_time: Optional[float] = None
        previous_end_position = initial_position.copy()
        previous_end_rpy = initial_rpy.copy()
        publish_period = 1.0 / config.action_frequency_hz
        next_publish_time = time.monotonic()

        try:
            for chunk_idx in range(config.num_chunks):
                sleep_duration = next_publish_time - time.monotonic()
                if sleep_duration > 0.0:
                    time.sleep(sleep_duration)

                chunk = _random_chunk(
                    now=time.monotonic(),
                    previous_end_time=previous_end_time,
                    previous_end_position=previous_end_position,
                    previous_end_rpy=previous_end_rpy,
                    rng=rng,
                    config=config,
                )
                interpolator.enqueue_chunk(chunk)
                action_waypoints.append(chunk[:, :3].copy())

                previous_end_time = float(chunk[-1, 6])
                previous_end_position = chunk[-1, :3].copy()
                previous_end_rpy = chunk[-1, 3:6].copy()
                next_publish_time += publish_period

            end_time_holder["final_end_time"] = previous_end_time
        except BaseException as exc:
            publish_error.append(exc)
        finally:
            publish_done.set()

    interpolator_thread = threading.Thread(
        target=run_interpolator,
        name="interpolator-runner",
        daemon=True,
    )
    publisher_thread = threading.Thread(
        target=publish_chunks,
        name="interpolator-publisher",
        daemon=True,
    )

    interpolator_thread.start()
    publisher_thread.start()

    poll_period = 1.0 / config.poll_frequency_hz
    try:
        while True:
            if publish_error:
                raise publish_error[0]
            if thread_error:
                raise thread_error[0]

            position = _read_position_key(redis_client, DEFAULT_POSITION_KEY)
            if position is not None:
                logged_positions.append(position)

            final_end_time = end_time_holder["final_end_time"]
            if publish_done.is_set() and final_end_time is not None:
                if time.monotonic() >= final_end_time + config.tail_duration:
                    break

            time.sleep(poll_period)
    finally:
        interpolator.stop()
        publisher_thread.join(timeout=1.0)
        interpolator_thread.join(timeout=1.0)
        redis_client.delete(DEFAULT_POSITION_KEY, DEFAULT_ORIENTATION_KEY)

    if publish_error:
        raise publish_error[0]
    if thread_error:
        raise thread_error[0]
    if not logged_positions:
        raise RuntimeError(
            "No interpolated samples were read back from Redis. "
            "Check that the Redis server is running and writable."
        )

    print(
        "Completed visual interpolator test with "
        f"{config.num_chunks} chunks and {len(logged_positions)} logged samples."
    )
    _plot_results(
        config=config,
        action_waypoints=action_waypoints,
        logged_positions=np.vstack(logged_positions),
    )


if __name__ == "__main__":
    try:
        main()
    except redis.RedisError as exc:
        raise SystemExit(
            "Redis connection failed. Start Redis and retry this visual test.\n"
            f"Details: {exc}"
        ) from exc
    except InterpolatorFault as exc:
        raise SystemExit(
            "The interpolator faulted during the visual test. "
            "Try lowering the action chunk rate or overlap.\n"
            f"Details: {exc}"
        ) from exc
