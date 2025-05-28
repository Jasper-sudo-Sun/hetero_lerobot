import os
import cv2
import torch
import pandas as pd
import torchvision.transforms.functional as TF
import numpy as np 

from lerobot.common.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import torch.nn.functional as F

checkpoint_dir = "/share/project/lvhuaihai/lvhuaihai/hetero_lerobot/outputs/train/pika/act/pika_pick_peach_without_ego_feat_20250528_015121/checkpoints/090000/pretrained_model"

# observation example:
# observation = {
#         "observation.images.cam_left_wrist": cam_left.to('cuda'),
#         "observation.images.cam_right_wrist": cam_right.to('cuda'),
#         "observation.images.cam_left_wrist_fisheye": cam_left_fisheye.to('cuda'),
#         "observation.images.cam_right_wrist_fisheye": cam_right_fisheye.to('cuda'),
#         "prompt": "stack the brown basket on the black basket",
#         "repo_id": "HuaihaiLyu/pika_pick_peach_without_ego_feat"
#     }

def get_obs():
    # TODO: LiMing Chen
    return

def action_publish():
    # TODO: QianPu Sun
    return


if __name__ == "__main__":
    # repo_id doesn't matter.
    dataset_meta = LeRobotDatasetMetadata(repo_id="HuaihaiLyu/pika_pick_peach_without_ego_feat", root='/share/project/lvhuaihai/robot_data/lerobot/HuaihaiLyu/pika_pick_peach_without_ego_feat')

    kwargs = {}
    kwargs["pretrained_name_or_path"] = checkpoint_dir
    kwargs["dataset_stats"] = dataset_meta
    policy_cfg = PreTrainedConfig.from_pretrained(**kwargs)
    policy_cfg.pretrained_path = checkpoint_dir
    policy = make_policy(policy_cfg, ds_meta=dataset_meta).to('cuda')

    
    while True:
        with torch.no_grad():
            observation = get_obs()
            pred_action = policy.forward_action(observation).squeeze()
            action_publish(pred_action)
            import pdb
            pdb.set_trace()

    