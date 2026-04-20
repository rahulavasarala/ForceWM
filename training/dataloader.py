import torch
from torch.utils.data import Dataset, DataLoader
import bisect
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path

from parquet_utils import ParquetDatasetReader, VideoDatasetReader



class MultiModalDataset(Dataset):
    def __init__(self, dataset_path):

        self.lowdim_keys = {}
        dataset_path = Path(dataset_path)

        metadata_path = dataset_path / "metadata.npz"
        metadata = np.load(metadata_path)
        self.episode_ends = metadata["episode_ends"]

        parquet_path = dataset_path / "dummy.parquet"
        self.parquet_reader = ParquetDatasetReader(parquet_path)

        video_path = dataset_path / "videos"
        self.video_reader = VideoDatasetReader(video_path, metadata, cache_size = 4)

        self.__create_key_information()


    def __create_key_information(self):

        # creates the key information --- downsampling steps --- etc ---

        self.lowdim_keys["eef_pos"] = {"dss": 3, "obs_horizon": 3}
        self.lowdim_keys["eef_ori"] = {"dss" : 3, "obs_horizon": 3}
        self.action_info = {"dss": 3, "obs_horizon": 3}

    def get_episode(self,idx):
        return bisect.bisect_left(self.episode_ends, idx)

    def __len__(self):
        return self.episode_ends[-1] + 1

    def __get_item__(self, idx):

        #Find the relevant starts and ends of the episodes --- 
        episode_num = self.get_episode(idx)
        episode_end = self.episode_ends[episode_num]
        episode_start = 0 if episode_num == 0 else self.episode_ends[episode_num -1] + 1

        #Find the first and last indexes of the retrieval for the lowdim keys
        obs_length = self.lowdim_keys["eef_pos"]["dss"]*(self.lowdim_keys["eef_pos"]["obs_horizon"] -1)
        action_length = self.action_info["dss"]*(self.action_info["obs_horizon"] -1)
        first_index = idx - obs_length
        last_index = idx + action_length
        first_index = max(episode_start, first_index)
        last_index = min(episode_end, last_index)

        frame = self.parquet_reader.get_idx_range(first_index, last_index, self.lowdim_keys.keys())

        bias = first_index

        obs_indices = np.arange(idx - obs_length, idx +1, self.lowdim_keys["eef_pos"]["dss"])
        obs_indices[obs_indices <= first_index] = first_index # Padding the observations
        action_indices = np.arange(idx, idx + action_length +1, self.action_info["dss"])
        action_indices[action_indices > last_index] = last_index

        obs_indices -= bias
        action_indices -= bias

        obs_dict = {k : v[obs_indices] for k, v in frame.items()}
        action_dict = {k : v[action_indices] for k, v in frame.items()}
        #Just set up the video reader ---- which will be good

        video_dict = self.video_reader.get_idx_range(first_index, last_index)
        video_dict = video_dict[obs_indices]

        obs_dict["images"] = video_dict


        return {"obs": obs_dict, "actions": action_dict}

if __name__ == "__main__":
    dataset_path = "/Users/rahulavasarala/Desktop/ForceWM/training/dummy_dataset"
    test_dataset = MultiModalDataset(dataset_path)

    print(test_dataset.__get_item__(15))




