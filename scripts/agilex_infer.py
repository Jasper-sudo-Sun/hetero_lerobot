import os
import cv2
import torch
import pandas as pd
import torchvision.transforms.functional as TF
import numpy as np 
from scipy.spatial.transform import Rotation as R
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

def xyzrpy_to_mat(x, y, z, roll, pitch, yaw):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    T[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    return T

def mat_to_xyzrpy(T):
    x, y, z = T[:3, 3]
    roll, pitch, yaw = R.from_matrix(T[:3, :3]).as_euler('xyz')
    return [x, y, z, roll, pitch, yaw]

def relative_to_absolute_poses(pred_action, abs_pose):
    """
    pred_action: (100, 6) list of relative [dx, dy, dz, droll, dpitch, dyaw]
    abs_pose: [x, y, z, roll, pitch, yaw] initial absolute pose in base frame
    return: (100, 6) list of absolute poses in base frame
    """
    abs_poses = []
    T_abs = xyzrpy_to_mat(*abs_pose)
    abs_poses.append(abs_pose)

    for i in range(pred_action.shape[0]):
        delta = pred_action[i].tolist()
        T_delta = xyzrpy_to_mat(*delta)
        T_abs = T_abs @ T_delta  # transform delta into base frame
        abs_pose = mat_to_xyzrpy(T_abs)
        abs_poses.append(abs_pose)

    return np.array(abs_poses)  # shape: (101, 6)

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

    