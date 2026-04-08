from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import redis


DEFAULT_CENTER = np.array([0.3, 0.0, 0.3], dtype=float)
DEFAULT_NUM_SAMPLES = 5
DEFAULT_POSITION_SPREAD = 0.05
DEFAULT_SETTLE_TIME_S = 5.0


@dataclass(frozen=True)
class RedisKeyPair:
    desired_position: str
    current_position: str


@dataclass(frozen=True)
class ValidationResult:
    sample_index: int
    desired_position: np.ndarray
    measured_position: np.ndarray
    error_vector: np.ndarray
    error_norm: float


KEY_CANDIDATES = (
    RedisKeyPair(
        desired_position="sim::franka::desired_cartesian_position",
        current_position="sim::franka::current_cartesian_position",
    ),
    RedisKeyPair(
        desired_position="sai::sim::franka::desired_cartesian_position",
        current_position="sai::sim::franka::current_cartesian_position",
    ),
)


def _redis_text(value: bytes | str | None) -> str:
    if value is None:
        raise RuntimeError("Requested Redis key is missing.")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_vector(redis_client: redis.Redis, key: str) -> np.ndarray:
    raw_value = redis_client.get(key)
    vector = np.asarray(json.loads(_redis_text(raw_value)), dtype=float).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"Redis key `{key}` did not contain a 3D position vector.")
    return vector


def _write_vector(redis_client: redis.Redis, key: str, value: np.ndarray) -> None:
    redis_client.set(key, json.dumps(np.asarray(value, dtype=float).reshape(3).tolist()))


def _resolve_key_pair(redis_client: redis.Redis) -> RedisKeyPair:
    for candidate in KEY_CANDIDATES:
        if redis_client.exists(candidate.current_position):
            return candidate

    return KEY_CANDIDATES[0]


def _sample_targets(
    rng: np.random.Generator,
    center: np.ndarray,
    num_samples: int,
    spread: float,
) -> list[np.ndarray]:
    offsets = rng.uniform(-spread, spread, size=(num_samples, 3))
    return [center + offset for offset in offsets]


def ValidatePositionControl(
    redis_host: str = "127.0.0.1",
    redis_port: int = 6379,
    redis_db: int = 0,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    center: np.ndarray = DEFAULT_CENTER,
    spread: float = DEFAULT_POSITION_SPREAD,
    settle_time_s: float = DEFAULT_SETTLE_TIME_S,
    seed: int | None = None,
) -> list[ValidationResult]:
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        decode_responses=False,
    )
    redis_client.ping()

    key_pair = _resolve_key_pair(redis_client)
    center = np.asarray(center, dtype=float).reshape(3)
    rng = np.random.default_rng(seed)

    initial_position = _read_vector(redis_client, key_pair.current_position)
    print(f"Using desired position key: `{key_pair.desired_position}`")
    print(f"Using current position key: `{key_pair.current_position}`")
    print(f"Initial current position: {np.array2string(initial_position, precision=4)}")

    targets = _sample_targets(rng, center=center, num_samples=num_samples, spread=spread)
    results: list[ValidationResult] = []

    for sample_index, desired_position in enumerate(targets, start=1):
        print(
            f"\nSample {sample_index}/{num_samples}: sending desired position "
            f"{np.array2string(desired_position, precision=4)}"
        )
        _write_vector(redis_client, key_pair.desired_position, desired_position)
        time.sleep(settle_time_s)

        measured_position = _read_vector(redis_client, key_pair.current_position)
        error_vector = measured_position - desired_position
        error_norm = float(np.linalg.norm(error_vector))

        result = ValidationResult(
            sample_index=sample_index,
            desired_position=desired_position,
            measured_position=measured_position,
            error_vector=error_vector,
            error_norm=error_norm,
        )
        results.append(result)

        print(
            "Measured position: "
            f"{np.array2string(measured_position, precision=4)}"
        )
        print(
            "Position error: "
            f"{np.array2string(error_vector, precision=4)} "
            f"(norm = {error_norm:.6f} m)"
        )

    error_norms = np.array([result.error_norm for result in results], dtype=float)
    max_error_index = int(np.argmax(error_norms))
    worst_result = results[max_error_index]

    print("\nSummary")
    print(f"Mean position error: {float(np.mean(error_norms)):.6f} m")
    print(f"Max position error: {float(np.max(error_norms)):.6f} m")
    print(
        "Worst-case desired/measured pair: "
        f"{np.array2string(worst_result.desired_position, precision=4)} -> "
        f"{np.array2string(worst_result.measured_position, precision=4)}"
    )

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send random Cartesian end-effector targets through Redis and "
            "measure the resulting position tracking error."
        )
    )
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--spread", type=float, default=DEFAULT_POSITION_SPREAD)
    parser.add_argument("--settle-time", type=float, default=DEFAULT_SETTLE_TIME_S)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        default=DEFAULT_CENTER.tolist(),
        metavar=("X", "Y", "Z"),
        help="Center point for random target generation. Default: 0.0 0.3 0.5",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ValidatePositionControl(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        num_samples=args.samples,
        center=np.asarray(args.center, dtype=float),
        spread=args.spread,
        settle_time_s=args.settle_time,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
