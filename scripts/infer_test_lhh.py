import os
import cv2
import torch
import pandas as pd
import torchvision.transforms.functional as TF

from lerobot.common.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import torch.nn.functional as F

# 视频路径
cam_front_path = "/mnt/hpfs/baaiei/qianpusun/lerobot/data/HuaihaiLyu/agilex_groceries/videos/chunk-000/observation.images.cam_high/episode_000310.mp4"
cam_left_path = "/mnt/hpfs/baaiei/qianpusun/lerobot/data/HuaihaiLyu/agilex_groceries/videos/chunk-000/observation.images.cam_left_wrist/episode_000310.mp4"
cam_right_path = "/mnt/hpfs/baaiei/qianpusun/lerobot/data/HuaihaiLyu/agilex_groceries/videos/chunk-000/observation.images.cam_right_wrist/episode_000310.mp4"

# 状态数据路径
state_path = '/mnt/hpfs/baaiei/qianpusun/lerobot/data/HuaihaiLyu/agilex_groceries/data/chunk-000/episode_000310.parquet'

# 模型 checkpoint 路径
checkpoint_dir = "/mnt/hpfs/baaiei/qianpusun/lerobot/outputs/train/act/agilex_groceries_20250501_150401/checkpoints/200000/pretrained_model"


def extract_frame(video_path, frame_number=0):
    """从视频中读取指定帧图像"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[ERROR] 视频文件不存在: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"[ERROR] 无法打开视频文件: {video_path}")

    for _ in range(frame_number):
        cap.grab()
    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError(f"[ERROR] 无法读取第 {frame_number} 帧")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


def get_obs_n_frame(frame_num=0):
    """构造某一帧的 observation 数据"""
    cam_front = torch.from_numpy(extract_frame(cam_front_path, frame_num)).permute(2, 0, 1)
    cam_front = TF.resize(cam_front, size=[480, 640]).unsqueeze(0).float()

    cam_left = torch.from_numpy(extract_frame(cam_left_path, frame_num)).permute(2, 0, 1).unsqueeze(0).float()
    cam_right = torch.from_numpy(extract_frame(cam_right_path, frame_num)).permute(2, 0, 1).unsqueeze(0).float()

    # 加载 robot 状态
    df = pd.read_parquet(state_path)
    if frame_num >= len(df):
        raise IndexError(f"帧号 {frame_num} 超出 parquet 数据范围 (共 {len(df)} 帧)")
    state = df.iloc[frame_num]['observation.state'].astype("float32")
    state_tensor = torch.tensor(state).unsqueeze(0)
    action = torch.tensor(
        df.iloc[frame_num:frame_num+8]['observation.state'].tolist(),
        dtype=torch.float32
    )
    observation = {
        "observation.state": state_tensor.to('cuda'),
        "observation.images.cam_high": cam_front.to('cuda'),
        "observation.images.cam_left_wrist": cam_left.to('cuda'),
        "observation.images.cam_right_wrist": cam_right.to('cuda'),
        "prompt": "stack the brown basket on the black basket"
    }

    return observation, action


if __name__ == "__main__":
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_dir)
    dataset_meta = LeRobotDatasetMetadata(repo_id="HuaihaiLyu/agilex_groceries", root='/mnt/hpfs/baaiei/qianpusun/lerobot/data/HuaihaiLyu/agilex_groceries')
    import pdb
    pdb.set_trace()
    policy = make_policy(policy_cfg, ds_meta=dataset_meta)
    policy = policy.from_pretrained(config=policy_cfg, pretrained_name_or_path=checkpoint_dir).to('cuda')
    save_action = []
    for i in range(0, 500):  # 推理前5帧
        print(f"\n[Frame {i}] ====================")
        observation, next_action = get_obs_n_frame(i)
        next_action = next_action.cuda()
        with torch.no_grad():
            action = policy.forward_action(observation).squeeze(0)
    
        # print(action.shape)
        # print(next_action.shape)
        # print("Predicted action:", action)
        # print("Actual action:", next_action)
        # 每帧的 MSE: 结果为 (n_frame,)
        print(action.shape)
        print(next_action.shape)
        import pdb
        pdb.set_trace()
        framewise_mse = F.mse_loss(action[:8], next_action, reduction='none').mean(dim=1)
        print(f"{i}-th Framewise MSE:", framewise_mse)

        # print(action.shape)
        # save_action.append(action.cpu().numpy())  

    # import numpy as np
    # np.save(f"/mnt/hpfs/baaiei/qianpusun/lerobot/save_action/action.npy", save_action)

        # # 差分处理：从绝对状态改为相对 delta 状态（e.g., t1-t0, t2-t1,...）
        # gt_delta = next_action[1:] - next_action[:-1]
        # pred_delta = action[1:] - action[:-1]
        
        # # 每帧 MSE (length: n_frame - 1)
        # dframewise_mse = F.mse_loss(pred_delta, gt_delta, reduction='none').mean(dim=1)

        # print(f"{i}-th Framewise ΔMSE:", dframewise_mse)
        # import pdb
        # pdb.set_trace()
        

# python scripts/infer_test.py --dataset.repo_id=data/mixed_groceries_bag --policy.type="diffusion"
# python scripts/act_infer.py --dataset.repo_id=data/mixed_groceries_bag --policy.type="act"