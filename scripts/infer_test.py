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
cam_front_path = "/home/sunqianpu/share_project/repo/lerobot/data/HuaihaiLyu/agilex_groceries_400/videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
# cam_front_path = "/mnt/hpfs/baaiei/chenliming/data/lerobot_data/realman/groceries_bag/videos/chunk-000/observation.images.cam_high/episode_000000.mp4"

cam_left_path = "/home/sunqianpu/share_project/repo/lerobot/data/HuaihaiLyu/agilex_groceries_400/videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4"
# cam_left_path = "/mnt/hpfs/baaiei/chenliming/data/lerobot_data/realman/groceries_bag/videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4"

cam_right_path = "/home/sunqianpu/share_project/repo/lerobot/data/HuaihaiLyu/agilex_groceries_400/videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4"
# cam_right_path = "/mnt/hpfs/baaiei/chenliming/data/lerobot_data/realman/groceries_bag/videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4"
# 状态数据路径
state_path = '/home/sunqianpu/share_project/data/lerobot_data/HuaihaiLyu/agilex_groceries_400/data/chunk-000/episode_000000.parquet'
# state_path = '/mnt/hpfs/baaiei/chenliming/data/lerobot_data/realman/pour/data/chunk-000/episode_000000.parquet'

# 模型 checkpoint 路径
checkpoint_dir = "/home/sunqianpu/share_project/ckpts/train/act/20250512_004358_agilex_pika_groceries_400_decoder_7/checkpoints/100000/pretrained_model"
# checkpoint_dir = "/home/sunqianpu/share_project/ckpts/train/act/20250512_005246_agilex_groceries_400_decoder_7_nrealtive/checkpoints/150000/pretrained_model"
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
    cam_front = torch.from_numpy(extract_frame(cam_front_path, frame_num)).permute(2, 0, 1) / 255
    cam_front = TF.resize(cam_front, size=[480, 640]).unsqueeze(0).float()

    cam_left = torch.from_numpy(extract_frame(cam_left_path, frame_num)).permute(2, 0, 1).unsqueeze(0).float() / 255
    cam_right = torch.from_numpy(extract_frame(cam_right_path, frame_num)).permute(2, 0, 1).unsqueeze(0).float() / 255
    
    # cam_front, cam_left, cam_right = get_obs_from_photo()
    # print(cam_front.shape, cam_left.shape, cam_right.shape)
    # 加载 robot 状态
    df = pd.read_parquet(state_path)
    if frame_num >= len(df):
        raise IndexError(f"帧号 {frame_num} 超出 parquet 数据范围 (共 {len(df)} 帧)")
    state = df.iloc[frame_num]['observation.state'].astype("float32")
    state_tensor = torch.tensor(state).unsqueeze(0)

    # state_tensor = torch.tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00, -1.0000e-04,  5.6127e-02,
    #       0.0000e+00,  2.1327e-01,  0.0000e+00,  1.4835e+00,  0.0000e+00,
    #       5.6127e-02,  0.0000e+00,  2.1327e-01,  0.0000e+00,  1.4835e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
    #       0.0000e+00,  0.0000e+00,  0.0000e+00]]).to('cuda')
    # state_tensor = torch.tensor([[-0.2007,  2.0241, -1.2941,  0.0287,  0.4308,  0.0984,  0.0152,  0.0166,
    #       1.2046, -0.8216,  0.1498,  0.1825, -0.1000,  0.0466,  0.4077, -0.0819,
    #       0.1470,  3.0625,  0.4911,  2.7995,  0.2621,  0.0068,  0.3067,  3.0233,
    #       1.0909,  3.0556,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]

    # state_tensor = torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    #      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0561, 0.0000, 0.2133, 0.0000,
    #      1.4835, 0.0000, 0.0561, 0.0000, 0.2133, 0.0000, 1.4835, 0.0000, 0.0000,
    #      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    #      0.0000, 0.0000]], device='cuda:0')

    action = torch.tensor(
        df.iloc[frame_num:frame_num+100]['observation.state'].tolist(),
        dtype=torch.float32
    )
    observation = {
        "observation.state": state_tensor.to('cuda'),
        "observation.images.cam_high": cam_front.to('cuda'),
        "observation.images.cam_left_wrist": cam_left.to('cuda'),
        "observation.images.cam_right_wrist": cam_right.to('cuda'),
        "prompt": "stack the brown basket on the black basket",
        "repo_id": "HuaihaiLyu/agilex_groceries_400"
    }

    return observation, action

# high_cam_path = "/mnt/hpfs/baaiei/qianpusun/lerobot/scripts/9951746172315_.pic.jpg"
# left_cam_path = "/mnt/hpfs/baaiei/qianpusun/lerobot/scripts/9961746172315_.pic.jpg"
# right_cam_path = "/mnt/hpfs/baaiei/qianpusun/lerobot/scripts/9971746172315_.pic.jpg"

# def get_obs_from_photo():

#     cam_high_photo = torch.from_numpy(cv2.cvtColor(cv2.imread(high_cam_path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float() / 255
#     cam_high_photo = TF.resize(cam_high_photo, size=[480, 640])

#     cam_left_photo = torch.from_numpy(cv2.cvtColor(cv2.imread(left_cam_path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float() / 255

#     cam_right_photo = torch.from_numpy(cv2.cvtColor(cv2.imread(right_cam_path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float() / 255
    
#     return cam_high_photo, cam_left_photo, cam_right_photo

if __name__ == "__main__":
    dataset_meta = LeRobotDatasetMetadata(repo_id="HuaihaiLyu/agilex_groceries_400", root='/home/sunqianpu/share_project/data/lerobot_data/HuaihaiLyu/agilex_groceries_400')
    # dataset_meta = LeRobotDatasetMetadata(repo_id="realman/groceries_bag", root='/mnt/hpfs/baaiei/chenliming/data/lerobot_data/realman/groceries_bag')
    kwargs = {}
    kwargs["pretrained_name_or_path"] = checkpoint_dir
    kwargs["dataset_stats"] = dataset_meta
    policy_cfg = PreTrainedConfig.from_pretrained(**kwargs)
    policy_cfg.pretrained_path = checkpoint_dir
    policy = make_policy(policy_cfg, ds_meta=dataset_meta).to('cuda')

    for j in range(3):
        pred_actions = []
        observation, next_action = get_obs_n_frame(j*100)
        for i in range(0, 100):  # 推理前5帧
            # print(f"\n[Frame {i}] ====================")
            gt_action = next_action.cuda()
            with torch.no_grad():
                pred_action = policy.select_action(observation)
            print(pred_action.shape)
            pred_actions.append(pred_action)
        pred_actions = torch.cat(pred_actions, dim=0).cuda()

        print(pred_actions.shape, gt_action.shape)
        print("save_action", pred_actions[:, :26])
        print("next_action", gt_action[:, :26])
        import numpy as np 

        mse = F.mse_loss(pred_actions, gt_action, reduction='none').mean(dim=1)
        print(mse)

        np.save(f"/home/sunqianpu/share_project/repo/lerobot/save_action/agilex_pred_action{j*100}.npy", pred_actions.cpu().numpy())
        np.save(f"/home/sunqianpu/share_project/repo/lerobot/save_action/agilex_gt_action{j*100}.npy", gt_action.cpu().numpy())
        # np.save(f"/mnt/hpfs/baaiei/qianpusun/lerobot/save_action/gt_action{j*100}.npy", next_action.cpu().numpy())

    # for i in range(len(mse)):
    #     print("save_action", save_action[i])
    #     print("next_action", next_action[i])

        # print(action.shape)
        # print(next_action.shape)
        # print("Predicted action:", action)
        # print("Actual action:", next_action)
        # 每帧的 MSE: 结果为 (n_frame,)
        # print(action.shape)
        # framewise_mse = F.mse_loss(action, next_action, reduction='none').mean(dim=1)
        # print(f"{i}-th Framewise MSE:", framewise_mse)

        # print(action.shape)
        # save_action.append(action.cpu().numpy())  


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