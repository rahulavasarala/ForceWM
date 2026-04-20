import bisect
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def convert_to_numpy_dict(row_group: pa.Table | dict) -> dict[str, np.ndarray]:
    if isinstance(row_group, dict):
        return {
            key: np.stack(value, axis=0) if len(value) and isinstance(value[0], (list, tuple, np.ndarray)) else np.asarray(value)
            for key, value in row_group.items()
        }

    numpy_dict: dict[str, np.ndarray] = {}
    for col in row_group.column_names:
        values = row_group[col].combine_chunks().to_numpy(zero_copy_only=False)
        if values.dtype == object and len(values) and isinstance(values[0], (list, tuple, np.ndarray)):
            values = np.stack(values, axis=0)
        numpy_dict[col] = values

    return numpy_dict

class ParquetDatasetReader: 

    def __init__(self, parquet_path):

        self.pf = pq.ParquetFile(parquet_path)


    def check_idx_oob(self, idx: int) -> None:
        num_rows = self.pf.metadata.num_rows
        if idx < 0 or idx >= num_rows:
            raise IndexError(f"Index {idx} is out of bounds for parquet with {num_rows} rows")


    def get_idx_range(self, idx_1: int, idx_2: int, cols: list[str] | None = None) -> pa.Table:
        self.check_idx_oob(idx_1)
        self.check_idx_oob(idx_2)

        if idx_1 > idx_2:
            raise ValueError("idx_1 cannot be greater than idx_2")

        columns = list(cols) if cols is not None else None
        slices: list[pa.Table] = []
        row_start = 0

        for row_group_index in range(self.pf.num_row_groups):
            row_group_rows = self.pf.metadata.row_group(row_group_index).num_rows
            row_end = row_start + row_group_rows - 1

            if row_end < idx_1:
                row_start += row_group_rows
                continue

            if row_start > idx_2:
                break

            local_start = max(idx_1, row_start) - row_start
            local_end = min(idx_2, row_end) - row_start
            length = local_end - local_start + 1

            table = self.pf.read_row_group(row_group_index, columns=columns)
            slices.append(table.slice(local_start, length))
            row_start += row_group_rows

        if not slices:
            selected_columns = columns or self.pf.schema.names
            return pa.table({column: pa.array([]) for column in selected_columns})

        slice = slices[0] if len(slices) == 1 else pa.concat_tables(slices)

        return convert_to_numpy_dict(slice)
    

class VideoDatasetReader:

    # The structure of the video dataset is the following ---- you have first the episode directory ---
    # And then you have the .mp4  

    # folder/ episode_num/ chunk_0001.mp4, chunk_0002.mp4, chunk_0003.mp4, chunk_0004.mp4
    # Running with the code ---- 

    #Chunk size is the size of each mp4 file in each episode directory

    def __init__(self, video_dataset_path, metadata, cache_size: int = 4):
        try:
            from decord import VideoReader
        except ImportError as exc:
            raise ImportError("VideoDatasetReader requires decord to be installed") from exc

        if "episode_ends" not in metadata or "chunk_size" not in metadata:
            raise ValueError("metadata must contain episode_ends and chunk_size")

        self.video_dataset_path = Path(video_dataset_path)
        self.episode_ends = np.asarray(metadata["episode_ends"], dtype=np.int64)
        self.episode_ends_list = self.episode_ends.tolist()
        self.chunk_size = int(np.asarray(metadata["chunk_size"]).item())
        self.cache_size = max(1, int(cache_size))
        self._video_reader_cls = VideoReader
        self._reader_cache: OrderedDict[Path, VideoReader] = OrderedDict()
        self._episode_chunk_files: dict[int, list[Path]] = {}

        if self.episode_ends.ndim != 1 or len(self.episode_ends) == 0:
            raise ValueError("episode_ends must be a non-empty 1D array")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    def get_idx_range(self, idx_1: int, idx_2: int):
        self.check_idx_oob(idx_1)
        self.check_idx_oob(idx_2)

        if idx_1 > idx_2:
            raise ValueError("idx_1 cannot be greater than idx_2")

        frame_count = idx_2 - idx_1 + 1
        frame_locations: dict[Path, list[tuple[int, int]]] = {}

        for position, global_idx in enumerate(range(idx_1, idx_2 + 1)):
            chunk_path, local_frame_idx = self._resolve_frame_location(global_idx)
            if chunk_path not in frame_locations:
                frame_locations[chunk_path] = []
            frame_locations[chunk_path].append((position, local_frame_idx))

        frames: list[np.ndarray | None] = [None] * frame_count
        for chunk_path, locations in frame_locations.items():
            reader = self._get_reader(chunk_path)
            local_indices = [local_idx for _, local_idx in locations]
            batch = reader.get_batch(local_indices).asnumpy()

            for batch_index, (position, _) in enumerate(locations):
                frames[position] = batch[batch_index]

        return np.stack(frames, axis=0)

    def get_frame(self, idx: int) -> np.ndarray:
        return self.get_idx_range(idx, idx)[0]

    def check_idx_oob(self, idx: int) -> None:
        num_rows = int(self.episode_ends[-1]) + 1
        if idx < 0 or idx >= num_rows:
            raise IndexError(f"Index {idx} is out of bounds for video dataset with {num_rows} frames")

    def __len__(self) -> int:
        return int(self.episode_ends[-1]) + 1

    def _resolve_frame_location(self, idx: int) -> tuple[Path, int]:
        episode_idx = bisect.bisect_left(self.episode_ends_list, idx)
        episode_start = 0 if episode_idx == 0 else int(self.episode_ends[episode_idx - 1]) + 1
        frame_in_episode = idx - episode_start
        chunk_idx = frame_in_episode // self.chunk_size
        local_frame_idx = frame_in_episode % self.chunk_size

        chunk_files = self._get_episode_chunk_files(episode_idx)
        if chunk_idx >= len(chunk_files):
            raise IndexError(
                f"Frame {idx} maps to missing chunk {chunk_idx} in episode_{episode_idx}"
            )

        return chunk_files[chunk_idx], local_frame_idx

    def _get_episode_chunk_files(self, episode_idx: int) -> list[Path]:
        cached_chunk_files = self._episode_chunk_files.get(episode_idx)
        if cached_chunk_files is not None:
            return cached_chunk_files

        episode_path = self.video_dataset_path / f"episode_{episode_idx}"
        if not episode_path.exists():
            raise FileNotFoundError(f"Missing episode directory: {episode_path}")

        chunk_files = sorted(episode_path.glob("chunk_*.mp4"))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {episode_path}")

        self._episode_chunk_files[episode_idx] = chunk_files
        return chunk_files

    def _get_reader(self, chunk_path: Path):
        cached_reader = self._reader_cache.pop(chunk_path, None)
        if cached_reader is not None:
            self._reader_cache[chunk_path] = cached_reader
            return cached_reader

        reader = self._video_reader_cls(str(chunk_path))
        self._reader_cache[chunk_path] = reader

        if len(self._reader_cache) > self.cache_size:
            self._reader_cache.popitem(last=False)

        return reader

    
