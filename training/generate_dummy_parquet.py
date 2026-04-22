import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

path = Path("./dummy_dataset/metadata.npz")
path.parent.mkdir(parents = True, exist_ok = True)
video_dataset_path = Path("./dummy_dataset/videos")

DATASET_LEN = 650


def generate_parquet_data(): 

    eef_pos = np.zeros((DATASET_LEN, 3))

    eef_pos_array = pa.FixedSizeListArray.from_arrays(
        pa.array(eef_pos.reshape(-1), type=pa.float32()),
        3
    )

    eef_ori = np.eye(3)

    eef_ori = np.vstack([eef_ori]* DATASET_LEN)
    eef_ori = np.reshape(eef_ori, (DATASET_LEN, 9))

    eef_ori_array = pa.FixedSizeListArray.from_arrays(
        pa.array(eef_ori.reshape(-1), type=pa.float32()),
        9
    )


    data = {
        "eef_pos": eef_pos_array,
        "eef_ori": eef_ori_array
    }

    table = pa.table(data)
    print("Writing the table with the specified data above!")
    pq.write_table(table, "./dummy_dataset/dummy.parquet", row_group_size=1000)
    print("Wrote the data in dummy.parquet!")

def generate_metadata(chunk_size: int):

    episode_ends = np.arange(20,  601, 20)
    metadata = {"episode_ends" : episode_ends, "chunk_size": np.asarray(chunk_size, dtype=np.int64)}
    metadata["chunk_size"] = chunk_size
    np.savez(str(path), **metadata)

    return episode_ends

def generate_video_data(episode_ends, chunk_size):

    if cv2 is None:
        raise ImportError("generate_video_data requires opencv-python to be installed")

    frame_height = 224
    frame_width = 224
    fps = 10
    black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_dataset_path.mkdir(parents=True, exist_ok=True)

    for episode_idx, episode_end in enumerate(episode_ends):
        episode_start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1]) + 1
        episode_length = int(episode_end) - episode_start + 1
        episode_path = video_dataset_path / f"episode_{episode_idx}"
        episode_path.mkdir(parents=True, exist_ok=True)

        num_chunks = (episode_length + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            frames_in_chunk = min(chunk_size, episode_length - chunk_start)
            chunk_path = episode_path / f"chunk_{chunk_idx + 1:04d}.mp4"

            writer = cv2.VideoWriter(str(chunk_path), fourcc, fps, (frame_width, frame_height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {chunk_path}")

            for _ in range(frames_in_chunk):
                writer.write(black_frame)

            writer.release()


episode_ends = generate_metadata(4)
generate_parquet_data()
generate_video_data(episode_ends, 4)

#Cool, this should be enough to generate the dummy data -- which is really good ---




