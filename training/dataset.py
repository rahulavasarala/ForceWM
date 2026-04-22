import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
import bisect
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path

import yaml

from parquet_utils import ParquetDatasetReader, VideoDatasetReader
from scipy.spatial.transform import Rotation as R

class MultiModalDataset(Dataset):
    def __init__(self, dataset_path, universal_contract):

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

        with open(universal_contract, "r") as f:
            contract = yaml.safe_load(f)

        self.action_type = contract.get("robot", {}).get("action", {}).get("mode", "relative")
        self.device = self.get_device()
        self.crop_size = None
        self.angle = None


    def __create_key_information(self):

        # creates the key information --- downsampling steps --- etc ---

        self.lowdim_keys["eef_pos"] = {"dss": 3, "obs_horizon": 3}
        self.lowdim_keys["eef_ori"] = {"dss" : 3, "obs_horizon": 3}
        self.action_info = {"dss": 3, "obs_horizon": 3}

    def get_episode(self,idx):
        return bisect.bisect_left(self.episode_ends, idx)

    def __len__(self):
        return self.episode_ends[-1] + 1
    
    def image_transforms(self, image_transforms):
        self.image_transforms = image_transforms

    def __getitem__(self, idx):

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
        self.convert_ori_to_quat(obs_dict)
        #Just set up the video reader ---- which will be good

        video_dict = self.video_reader.get_idx_range(first_index, last_index)
        video_dict = video_dict[obs_indices]

        obs_dict["images"] = self.ensure_channel_first_images(video_dict)

        if self.crop_size is not None or self.angle is not None:
            obs_dict["images"] = self.perform_visual_transformations(obs_dict["images"])


        if self.action_type == "relative":
            action_dict = self.transform_action_to_relative(action_dict)

        action_dict = self.convert_to_torch(action_dict)
        obs_dict = self.convert_to_torch(obs_dict)

        return {"obs": obs_dict, "actions": action_dict}
    

    def transform_action_to_relative(self, action_dict):
        if "eef_pos" not in action_dict or "eef_ori" not in action_dict:
            raise KeyError("`action_dict` must contain `eef_pos` and `eef_ori`.")

        world_pos = np.asarray(action_dict["eef_pos"], dtype=np.float32).reshape(-1, 3)
        world_ori = np.asarray(action_dict["eef_ori"], dtype=np.float32).reshape(-1, 3, 3)

        base_frame = world_ori[0].copy()
        base_pos = world_pos[0].copy()
        world_to_base = base_frame.T

        relative_pos = (world_pos - base_pos) @ world_to_base.T
        relative_ori = np.einsum("ij,tjk->tik", world_to_base, world_ori)
        relative_ori_quat = R.from_matrix(relative_ori).as_quat().astype(np.float32)

        transformed_action_dict = dict(action_dict)
        transformed_action_dict["eef_pos"] = relative_pos.astype(np.float32)
        transformed_action_dict["eef_ori"] = relative_ori_quat

        return transformed_action_dict
    
    def convert_ori_to_quat(self, data_dict):

        ori = np.asarray(data_dict["eef_ori"], dtype=np.float32).reshape(-1, 3, 3)
        ori_quat = R.from_matrix(ori).as_quat().astype(np.float32)

        data_dict["eef_ori"] = ori_quat

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def convert_to_torch(self, data):
        if isinstance(data, dict):
            return {key: self.convert_to_torch(value) for key, value in data.items()}

        if isinstance(data, torch.Tensor):
            tensor = data
        else:
            tensor = torch.from_numpy(np.asarray(data))

        if torch.is_floating_point(tensor):
            tensor = tensor.float()

        return tensor.to(self.device)

    def ensure_channel_first_images(self, images):
        if isinstance(images, dict):
            return {key: self.ensure_channel_first_images(value) for key, value in images.items()}

        image_tensor = self.convert_to_torch(images)
        if not torch.is_floating_point(image_tensor):
            image_tensor = image_tensor.float() / 255.0
        if image_tensor.ndim == 4 and image_tensor.shape[-1] in (1, 3):
            return image_tensor.permute(0, 3, 1, 2).contiguous()
        if image_tensor.ndim == 5 and image_tensor.shape[-1] in (1, 3):
            return image_tensor.permute(0, 1, 4, 2, 3).contiguous()
        return image_tensor

    
    def perform_visual_transformations(self, video_dict):
        if isinstance(video_dict, dict):
            return {
                key: self.perform_visual_transformations(value)
                for key, value in video_dict.items()
            }

        image_stack = self.convert_to_torch(video_dict).detach().cpu().float()
        if image_stack.ndim != 4:
            raise ValueError(
                f"Video stack must have shape (T, C, H, W), got {tuple(image_stack.shape)}."
            )

        time_steps, channels, height, width = image_stack.shape
        if channels not in (1, 3):
            raise ValueError(
                f"Video stack must have 1 or 3 channels, got {channels}."
            )

        crop_size = self.crop_size if self.crop_size is not None else (height, width)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        crop_height, crop_width = crop_size

        crop_top, crop_left, crop_height, crop_width = RandomCrop.get_params(
            image_stack[0], output_size=(crop_height, crop_width)
        )

        rotation_angle = 0.0 if self.angle is None else float(self.angle)
        transformed_frames = []
        for frame_index in range(time_steps):
            frame = image_stack[frame_index]
            frame = TF.crop(frame, crop_top, crop_left, crop_height, crop_width)
            frame = TF.rotate(
                frame,
                rotation_angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
            )
            frame = TF.resize(
                frame,
                size=[height, width],
                interpolation=InterpolationMode.BILINEAR,
            )
            transformed_frames.append(frame)

        return torch.stack(transformed_frames, dim=0).to(self.device)


if __name__ == "__main__":
    dataset_path = "/Users/rahulavasarala/Desktop/ForceWM/training/dummy_dataset"
    contract_path = "/Users/rahulavasarala/Desktop/ForceWM/universal_contract.yaml"
    test_dataset = MultiModalDataset(dataset_path, universal_contract= contract_path)

    print(test_dataset.__getitem__(15))
