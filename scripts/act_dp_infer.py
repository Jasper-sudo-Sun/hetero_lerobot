
from lerobot.common.policies.factory import make_policy
from lerobot.configs.types import DictLike, FeatureType, PolicyFeature

import cv2
import os
# from agilex_env import get_obs
import torch
import torchvision.transforms.functional as TF
def extract_frame(video_path, save_path=None, frame_number=50):
    """
    读取视频的指定帧图像并保存（可选）

    Args:
        video_path (str): 视频文件的路径
        save_path (str or None): 保存指定帧图像的路径，如果为 None 则不保存
        frame_number (int): 要读取的帧号（从0开始计数）
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] 视频文件不存在: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频文件: {video_path}")
        return

    # 跳过前 frame_number - 1 帧
    for _ in range(frame_number - 1):
        success = cap.grab()
        if not success:
            print(f"[ERROR] 无法读取第 {_+1} 帧")
            cap.release()
            return

    # 读取第 frame_number 帧
    success, frame = cap.read()
    cap.release()

    if not success:
        print(f"[ERROR] 无法读取第 {frame_number} 帧")
        return

    print(f"[INFO] 成功读取第 {frame_number} 帧")

    if save_path:
        cv2.imwrite(save_path, frame)
        print(f"[INFO] 第 {frame_number} 帧已保存至: {save_path}")

    return frame


# config = config.get_config("pi0_aloha_stack_basket")
checkpoint_dir = "/mnt/hpfs/baaiei/qianpusun/lerobot/outputs/train/act_groceries_bag/checkpoints/500000/pretrained_model"

# Create a trained policy.

cam_front_path = "/mnt/hpfs/baaiei/lvhuaihai/episode_000000_high.mp4"
cam_left_path = "/mnt/hpfs/baaiei/lvhuaihai/episode_000000_left.mp4"
cam_right_path = "/mnt/hpfs/baaiei/lvhuaihai/episode_000000_right.mp4"

cam_front = torch.from_numpy(extract_frame(cam_front_path)).permute(2, 0, 1) 
cam_front = TF.resize(cam_front, size=[480, 640]).unsqueeze(0)
cam_left = torch.from_numpy(extract_frame(cam_left_path)).permute(2, 0, 1).unsqueeze(0)
cam_right = torch.from_numpy(extract_frame(cam_right_path)).permute(2, 0, 1).unsqueeze(0)


state = [0.006716111209243536, 0, -0.4529101252555847, -0.07096400111913681,
         1.0769851207733154, 0.10430033504962921, -0.00007000000186963007,
         -0.26855722069740295, 0, -0.4851300120353699, 0.19741877913475037,
         1.0484634637832642, -0.09125188738107681, 0.0002800000074785203,
         0,0,0,0,0,0,0,0,0,0,0,0]
# Run inference on a dummy example.
observation = {
    "observation.state": torch.tensor(state).to('cuda').unsqueeze(0),
    "observation.images.cam_high": cam_front.to('cuda'),
    "observation.images.cam_left_wrist": cam_left.to('cuda'),
    "observation.images.cam_right_wrist": cam_right.to('cuda'),
    "prompt": "stack the brown basket on the black basket"
}

from lerobot.common.policies.factory import make_policy
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.configs.policies import PreTrainedConfig

checkpoint_dir = "/mnt/hpfs/baaiei/qianpusun/lerobot/outputs/train/dp_groceries_bag/checkpoints/500000/pretrained_model"
def infer():

    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_dir)
    dataset_meta = LeRobotDatasetMetadata(
        "data/mixed_groceries_bag", root=None, revision=None
    )
    policy = make_policy(policy_cfg, ds_meta=dataset_meta)

    policy.from_pretrained(config=policy_cfg, pretrained_name_or_path=checkpoint_dir).to('cuda')

    action = policy.select_action(observation).squeeze(0)

    print("action", action)

if __name__ == "__main__":
    infer()



# python scirpts/act_infer.py --dataset.repo_id=data/mixed_groceries_bag --policy.type="diffusion"
# python scirpts/act_infer.py --dataset.repo_id=data/mixed_groceries_bag --policy.type="act"