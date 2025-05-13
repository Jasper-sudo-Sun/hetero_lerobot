#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import logging
import shutil
from pathlib import Path
from typing import Callable
from torchvision.transforms.functional import resize

import datasets
import numpy as np
import packaging.version
import PIL.Image
import torch
import torch.utils
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.common.datasets.image_writer import AsyncImageWriter, write_image
from lerobot.common.datasets.utils import (
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    INFO_PATH,
    TASKS_PATH,
    append_jsonlines,
    backward_compatible_episodes_stats,
    check_delta_timestamps,
    check_timestamps_sync,
    check_version_compatibility,
    create_empty_dataset_info,
    create_lerobot_dataset_card,
    embed_images,
    get_delta_indices,
    get_episode_data_index,
    get_features_from_robot,
    get_hf_features_from_features,
    get_safe_version,
    hf_transform_to_torch,
    is_valid_version,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_stats,
    load_tasks,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
    write_json,
)
from lerobot.common.datasets.video_utils import (
    VideoFrame,
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
    get_video_info,
)
from lerobot.common.robot_devices.robots.utils import Robot

CODEBASE_VERSION = "v2.1"




class HeteroMergedLerobotDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: dict, hetero_datasets: dict, use_relative_as: bool):
        self.datasets = datasets  # {"松宁": dataset1, "VisionPro": dataset2, "Pika": dataset3}

        self.total_frames = 0
        self.total_episodes = 0
        self.episode_data_index = {}

        self.meta = datasets.meta
        self.episode_data_index = datasets.episode_data_index
        self.total_frames += datasets.meta.total_frames
        self.total_episodes += datasets.meta.total_episodes
    
        self.hetero_datasets = hetero_datasets
        self.hetero_meta = {}
        self.hetero_episode_data_index = {}
        for repo_id, hetero_dataset in hetero_datasets.items():
            self.hetero_meta[repo_id] = hetero_dataset.meta
            self.hetero_episode_data_index[repo_id] = hetero_dataset.episode_data_index
            self.total_frames += hetero_dataset.meta.total_frames
            self.total_episodes += hetero_dataset.meta.total_episodes

        self.full_data_dict = {self.datasets.repo_id: self.datasets, **self.hetero_datasets}
        self.use_relative_as = use_relative_as
        self.index_to_repo_and_idx = []
        for repo_id, dataset in self.full_data_dict.items():
            for idx in range(len(dataset)):
                self.index_to_repo_and_idx.append((repo_id, idx))
    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        return self.total_frames
    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return self.total_episodes

    def __len__(self):
        return len(self.index_to_repo_and_idx)
    
    def to_relative(self, action):
        # data shape: (T, 38)
        relative = np.zeros_like(action)
        relative[1:] = action[1:] - action[:-1]
        # 第一个时刻没有前一个时刻，可以保留为0或复制下一帧
        # relative[0] = data[1] - data[0]  # 可选
        return relative
    
    def __getitem__(self, idx):
        repo_id, real_idx = self.index_to_repo_and_idx[idx]
        sample = self.full_data_dict[repo_id][real_idx]
        sample["repo_id"] = repo_id

        # TODO(sqp): hard_code
        if "observation.images.cam_left_wrist" not in sample:
            sample["observation.images.cam_left_wrist"] = torch.zeros_like(sample["observation.images.cam_high"]).to(sample["observation.images.cam_high"].device)
        
        if "observation.images.cam_right_wrist" not in sample:
            sample["observation.images.cam_right_wrist"] = torch.zeros_like(sample["observation.images.cam_high"]).to(sample["observation.images.cam_high"].device)

        if "action" in sample and self.use_relative_as:
            relative_action = self.to_relative(sample["action"])
            sample["action"] = torch.from_numpy(relative_action).to(sample["observation.images.cam_high"].device)
        
        if "state" in sample and self.use_relative_as:
            relative_state = self.to_relative(sample["state"])
            sample["state"] = torch.from_numpy(relative_state).to(sample["observation.images.cam_high"].device)

        return sample