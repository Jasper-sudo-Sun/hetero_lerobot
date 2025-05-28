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
cam_left_path = "/share/project/lvhuaihai/robot_data/lerobot/HuaihaiLyu/pika_pick_peach_without_ego_feat/videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4"

cam_right_path = "/share/project/lvhuaihai/robot_data/lerobot/HuaihaiLyu/pika_pick_peach_without_ego_feat/videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4"

cam_left_fisheye_path = "/share/project/lvhuaihai/robot_data/lerobot/HuaihaiLyu/pika_pick_peach_without_ego_feat/videos/chunk-000/observation.images.cam_left_wrist_fisheye/episode_000000.mp4"

cam_right_fisheye_path = "/share/project/lvhuaihai/robot_data/lerobot/HuaihaiLyu/pika_pick_peach_without_ego_feat/videos/chunk-000/observation.images.cam_right_wrist_fisheye/episode_000000.mp4"

state_path = '/share/project/lvhuaihai/robot_data/lerobot/HuaihaiLyu/pika_pick_peach_without_ego_feat/data/chunk-000/episode_000000.parquet'

checkpoint_dir = "/share/project/lvhuaihai/lvhuaihai/hetero_lerobot/outputs/train/pika/act/pika_pick_peach_without_ego_feat_20250528_015121/checkpoints/090000/pretrained_model"

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
    # cam_front = torch.from_numpy(extract_frame(cam_front_path, frame_num)).permute(2, 0, 1) / 255
    # cam_front = TF.resize(cam_front, size=[480, 640]).unsqueeze(0).float()

    cam_left = torch.from_numpy(extract_frame(cam_left_path, frame_num)).permute(2, 0, 1).unsqueeze(0).float() / 255
    cam_right = torch.from_numpy(extract_frame(cam_right_path, frame_num)).permute(2, 0, 1).unsqueeze(0).float() / 255
    
    cam_left_fisheye = torch.from_numpy(extract_frame(cam_left_fisheye_path, frame_num)).permute(2, 0, 1).unsqueeze(0).float() / 255
    cam_right_fisheye = torch.from_numpy(extract_frame(cam_right_fisheye_path, frame_num)).permute(2, 0, 1).unsqueeze(0).float() / 255
    # print(cam_front.shape, cam_left.shape, cam_right.shape)
    # 加载 robot 状态
    df = pd.read_parquet(state_path)
    if frame_num >= len(df):
        raise IndexError(f"帧号 {frame_num} 超出 parquet 数据范围 (共 {len(df)} 帧)")
    # state = df.iloc[frame_num]['observation.state'].astype("float32")
    # state_tensor = torch.tensor(state).unsqueeze(0)
    action = torch.tensor(
        df.iloc[frame_num:frame_num+100]['action'].tolist(),
        dtype=torch.float32
    )
    observation = {
        # "observation.state": state_tensor.to('cuda'),
        # "observation.images.cam_high": cam_front.to('cuda'),
        "observation.images.cam_left_wrist": cam_left.to('cuda'),
        "observation.images.cam_right_wrist": cam_right.to('cuda'),
        "observation.images.cam_left_wrist_fisheye": cam_left_fisheye.to('cuda'),
        "observation.images.cam_right_wrist_fisheye": cam_right_fisheye.to('cuda'),
        "prompt": "stack the brown basket on the black basket",
        "repo_id": "HuaihaiLyu/pika_pick_peach_without_ego_feat"
    }

    return observation, action

def compute_mse_at_frame(frame_num):
    observation, next_action = get_obs_n_frame(frame_num)
    gt_action = next_action.cuda()

    with torch.no_grad():
        pred_action = policy.forward_action(observation).squeeze()

        # 主体动作维度 MSE（14~25）
        gt_sub = gt_action[:, 14:26]
        pred_sub = pred_action[:, 14:26]
        per_step_mse = F.mse_loss(pred_sub, gt_sub, reduction='none').mean(dim=1)

        # Gripper 动作维度 MSE（6 和 13）
        gt_gripper = gt_action[:, [6, 13]]
        pred_gripper = pred_action[:, [6, 13]]
        gripper_mse = F.mse_loss(pred_gripper, gt_gripper, reduction='none').mean(dim=1)

        # 全维度动作 MSE（0~37）
        total_mse = F.mse_loss(pred_action, gt_action, reduction='mean')

        return per_step_mse.cpu(), gripper_mse.cpu(), total_mse.item()

if __name__ == "__main__":

    dataset_meta = LeRobotDatasetMetadata(repo_id="HuaihaiLyu/pika_pick_peach_without_ego_feat", root='/share/project/lvhuaihai/robot_data/lerobot/HuaihaiLyu/pika_pick_peach_without_ego_feat')
    # dataset_meta = LeRobotDatasetMetadata(repo_id="HuaihaiLyu/agilex_groceries_400", root='/root/data/lerobot_data/HuaihaiLyu/agilex_groceries_400')

    kwargs = {}
    kwargs["pretrained_name_or_path"] = checkpoint_dir
    kwargs["dataset_stats"] = dataset_meta
    policy_cfg = PreTrainedConfig.from_pretrained(**kwargs)
    policy_cfg.pretrained_path = checkpoint_dir
    policy = make_policy(policy_cfg, ds_meta=dataset_meta).to('cuda')

    pred_actions = []
    observation, next_action = get_obs_n_frame(100)
    gt_action = next_action.cuda()

    # 帧列表
    frame_list = [100, 200, 300]

    for frame_id in frame_list:
        print(f"\n===== Frame {frame_id} =====")
        per_step_mse, gripper_mse, total_mse = compute_mse_at_frame(frame_id)
        print("Per-step MSE (14-25):", per_step_mse)
        print("Gripper MSE (6 & 13):", gripper_mse)
        print("Total MSE (0-37):", total_mse)

        import pdb
        pdb.set_trace()

    # pred_actions.append(pred_action)
        # pred_actions = torch.cat(pred_actions, dim=0).cuda()

        # print('pred_action', pred_actions.shape)
        # print('gt_action', gt_action.shape)
        # mask = torch.ones_like(pred_actions)
        # mask[:, 7] = 0  # 第七维，索引从0开始
        # mask[:, 15] = 0  # 第十五维，索引从0开始
        # pred_actions_masked = pred_actions * mask
        # gt_action_masked = gt_action * mask
        # mse = F.mse_loss(pred_actions_masked, gt_action_masked, reduction='none').mean(dim=1)
        # # print(mse)

        # np.save(f"/home/sunqianpu/share_project/repo/lerobot/outputs/save_action/realman_pred_action{j*100}.npy", pred_actions.cpu().numpy())
        # np.save(f"/home/sunqianpu/share_project/repo/lerobot/outputs/save_action/realman_gt_action{j*100}.npy", gt_action.cpu().numpy())
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