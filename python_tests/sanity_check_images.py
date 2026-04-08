from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import redis


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_output = script_dir / "sanity_check.mp4"

    parser = argparse.ArgumentParser(
        description="Capture JPEG-compressed sim frames from Redis and write them to an MP4."
    )
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-key", default="sim::franka::camera_01")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.1)
    parser.add_argument("--output-fps", type=float, default=10.0)
    parser.add_argument("--output-path", type=Path, default=default_output)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    redis_client = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=0,
        decode_responses=False,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer: cv2.VideoWriter | None = None
    frames_written = 0
    first_frame_shape: tuple[int, int] | None = None

    print(f"Capturing sim frames from `{args.redis_key}` for {args.duration_s:.1f}s.")
    print(f"Writing MP4 to: {output_path}")

    start_time = time.time()
    next_poll_time = time.perf_counter()
    while time.time() - start_time < args.duration_s:
        redis_bytes = redis_client.get(args.redis_key)
        if redis_bytes is not None:
            frame = cv2.imdecode(np.frombuffer(redis_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frame_size = (frame.shape[1], frame.shape[0])
                if video_writer is None:
                    video_writer = cv2.VideoWriter(
                        str(output_path),
                        fourcc,
                        float(args.output_fps),
                        frame_size,
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(f"Failed to open MP4 writer at {output_path}.")
                    first_frame_shape = frame_size
                elif frame_size != first_frame_shape:
                    frame = cv2.resize(frame, first_frame_shape)

                video_writer.write(frame)
                frames_written += 1

        next_poll_time += args.poll_interval_s
        sleep_duration = next_poll_time - time.perf_counter()
        if sleep_duration > 0.0:
            time.sleep(sleep_duration)
        else:
            next_poll_time = time.perf_counter()

    if video_writer is not None:
        video_writer.release()

    if frames_written == 0:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(
            f"No decodable frames were found at `{args.redis_key}` during the capture window."
        )

    print(f"Wrote {frames_written} frames to {output_path}")


if __name__ == "__main__":
    main()
