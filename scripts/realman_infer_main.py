# #为了replay加载.parquet文件导库
# import pandas as pd
# 导入策略工厂函数
from lerobot.common.policies.factory import make_policy
# 导入配置类型定义
from lerobot.configs.types import DictLike, FeatureType, PolicyFeature

import cv2
import os
# 从agilex环境导入观察函数(当前已注释)
# from agilex_env import get_obs
import torch
import torchvision.transforms.functional as TF
import logging
# 配置日志记录
logging.basicConfig(
        filename='/home/rm/repo/hetero_lerobot/scripts/lerobot_output.log',  # 日志文件路径
        filemode='w',  # 'a' 为追加模式，'w' 为覆盖模式
        format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
        level=logging.INFO  # 设置日志等级（DEBUG, INFO, WARNING, ERROR, CRITICAL）
    )
import numpy as np
from dora import Node
from PIL import Image
import pyarrow as pa
import time
from lerobot.common.policies.factory import make_policy
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.configs.policies import PreTrainedConfig
import pandas as pd
import h5py


def qpos_2_joint_positions(qpos:np.ndarray):
    """
    将原始关节位置数据转换为特定格式的关节位置
    
    参数:
        qpos: 原始关节位置数组
    
    返回:
        转换后的关节位置数组，包含左右手臂关节位置和夹爪位置
    """
    # 修改为7自由度，调整切片范围
    r_joint_pos = qpos[50:57]  # 修改为7自由度，从50到57提取7个值
    l_joint_pos = qpos[0:7]    # 修改为7自由度，从0到7提取7个值
    r_gripper_pos = np.array([qpos[60]])
    l_gripper_pos = np.array([qpos[10]])

    l_pos = np.concatenate((l_joint_pos,l_gripper_pos))
    r_pos = np.concatenate((r_joint_pos,r_gripper_pos))

    return np.concatenate((l_pos,r_pos))
    
# def save_image(image_array, key):
#     """
#     保存图像到指定路径，未使用
    
#     参数:
#         image_array: 图像数组
#         key: 用于创建目录和文件名的键
#     """
#     # 创建目录路径
#     dir_path = os.path.expanduser(f"~/repo/lerobot/infer_image/{key}")
#     os.makedirs(dir_path, exist_ok=True)

#     # 在对应文件夹中保存图像
#     image_path = os.path.join(dir_path, f"{key}_{int(torch.randint(0, 1000000, (1,)))}.png")
#     image = Image.fromarray(image_array)
#     image.save(image_path)
#     logging.info(f"Image saved to {image_path}")

def interpolate_actions(start_action, end_action, num_interpolations=1):
    """
    在两个动作之间进行线性插值
    
    参数:
        start_action: 起始动作（numpy数组）
        end_action: 结束动作（numpy数组）
        num_interpolations: 中间动作的数量（默认1）
    
    返回:
        包含起始和结束动作的插值动作列表
    """
    interpolated_actions = []
    for i in range(num_interpolations):
        alpha = i / float(num_interpolations)
        interpolated_action = (1 - alpha) * start_action + alpha * end_action
        interpolated_actions.append(interpolated_action)
    interpolated_actions.append(end_action)  # 确保包含最后一个动作
    return interpolated_actions

def main():
    """主函数，处理机器人控制流程"""
    # 创建Dora节点对象
    node = Node()
    # 初始化帧、关节和姿态字典
    frames = {}
    joints = {}
    pose = {}
    # 设置数据回放模式标志
    data_replay = False
    # 文件路径（已注释不同类型的文件路径选项）
    # # 当前使用的推理文件路径
    # file_path = '/home/rm/lerobot/outputs/train/realman_groceries_20250501_152526/save_action/action0_fixed_state_photo.npy'
    # 真实动作（已注释）
    file_path = '/home/rm/repo/hetero_lerobot/outputs/save_action/realman_pred_action0.npy'
    
    # parquet文件（已注释）
    # file_path = '/home/agilex/lerobot/outputs/train/agilex_groceries_20250501_150401/episode_000310.parquet'
    # file_path = '/home/rm/repo/lerobot/processed_data/realman/groceries_bag/data/chunk-000/episode_000010.parquet'
    
    # 加载文件数据
    df = np.load(file_path)
    # df = pd.read_parquet(file_path)
    # 根据模式执行不同的操作
    if data_replay == True:
        # 数据回放模式
        logging.info("数据回放模式")
        
        ##liming测试数据
        # 打印 DataFrame 的索引和列信息
        # print("到达到达！")
        # print("DataFrame index:", df.index)
        # print("DataFrame columns:", df.columns)
        
        # # 检查 DataFrame 是否为空
        # if df.empty:
        #     print("DataFrame is empty!")
        #     return
        
        # # 尝试访问数据
        # try:
        #     # 如果需要按位置访问第一行，使用 iloc
        #     action = df.iloc[0]
        #     print("First row:", action)
        # except Exception as e:
        #     print(f"Error accessing data: {e}")
            
            
        # if df.empty:
        #     print("DataFrame is empty!")
        #     return
        
        # # 打印 DataFrame 的索引和列信息
        # print("DataFrame index:", df.index)
        # print("DataFrame columns:", df.columns)
        
        # # 假设 i 是一个整数，用于访问行
        # i = 0  # 示例索引
        # if i < len(df):
        #     action = df.iloc[i]  # 按位置访问第 i 行
        #     print("Row at index", i, ":", action)
        # else:
        #     print(f"Index {i} is out of range for the DataFrame.")
        
        # # 如果需要访问特定列，确保使用正确的列名
        # column_name = 'action'
        # if column_name in df.columns:
        #     print(f"Column '{column_name}':", df[column_name])
        # else:
        #     print(f"Column '{column_name}' does not exist in the DataFrame.")
            
        ##liming测试数据
        
        
        for i in range(df.shape[0]): 
            action = df[i]
            # action = df[i]
            # 分离左右手臂动作，确保适配7自由度
            left_action = action[:8]   # 7自由度的左臂动作
            right_action = action[8:16]     # 7自由度的右臂动作
            logging.info(f"左臂动作{i}: {left_action}")
            logging.info(f"右臂动作{i}: {right_action}")
            # 发送关节状态到输出
            node.send_output("jointstate_left", pa.array(left_action.ravel()))
            node.send_output("jointstate_right", pa.array(right_action.ravel()))
            time.sleep(0.1)
    else:
        # 推理模式
        logging.info("进入推理模式")
        # 加载预训练模型检查点
        # checkpoint_dir = "/home/rm/repo/lerobot/outputs/train/act/realman_groceries_bag_20250501_152526/checkpoints/140000/pretrained_model"
        checkpoint_dir = "/home/rm/repo/hetero_lerobot/outputs/pretrained_model"
        # 创建数据集元数据
        # dataset_meta = LeRobotDatasetMetadata(repo_id='HuaihaiLyu', root='/home/rm/repo/lerobot/outputs/train/meta/realman_groceries')
        dataset_meta = LeRobotDatasetMetadata(repo_id='HuaihaiLyu/rm_groceries', root='/home/rm/repo/hetero_lerobot/outputs')
        kwargs = {}
        kwargs["pretrained_name_or_path"] = checkpoint_dir
        kwargs["dataset_stats"] = dataset_meta
        # 创建策略配置
        policy_cfg = PreTrainedConfig.from_pretrained(**kwargs)
        policy_cfg.pretrained_path = checkpoint_dir
        # 创建策略模型并移至GPU
        # print(policy_cfg.temporal_ensemble_coeff)
        policy = make_policy(policy_cfg, ds_meta=dataset_meta).to('cuda')
        
        logging.info("初始化策略")
        
        with torch.no_grad():
            # 处理节点事件
            for event in node:
                event_type = event["type"]
                if event_type == "INPUT":
                    event_id = event["id"]

                    # 处理图像输入
                    if "image" in event_id:
                        logging.info("image in")
                        storage = event["value"]
                        metadata = event["metadata"]
                        encoding = metadata["encoding"]

                        # 检查图像编码类型
                        if encoding == "bgr8" or encoding == "rgb8" or encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                            channels = 3
                            storage_type = np.uint8
                        else:
                            raise RuntimeError(f"不支持的图像编码: {encoding}")

                        # 根据不同的编码格式处理图像
                        if encoding == "bgr8":
                            width = metadata["width"]
                            height = metadata["height"]
                            frame = (
                                storage.to_numpy()
                                .astype(storage_type)
                                .reshape((height, width, channels))
                            )
                            frame = frame[:, :, ::-1]  # OpenCV图像（BGR转RGB）
                        elif encoding == "rgb8":
                            width = metadata["width"]
                            height = metadata["height"]
                            frame = (
                                storage.to_numpy()
                                .astype(storage_type)
                                .reshape((height, width, channels))
                            )
                        elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                            storage = storage.to_numpy()
                            frame = cv2.imdecode(storage, cv2.IMREAD_COLOR)
                            frame = frame[:, :, ::-1]  # OpenCV图像（BGR转RGB）
                        else:
                            raise RuntimeError(f"不支持的图像编码: {encoding}")
                        
                        # 存储图像帧
                        frames[f"last_{event_id}"] = frames.get(
                            event_id, Image.fromarray(frame),
                        )
                        frames[event_id] = Image.fromarray(frame) 

                        # 可选：保存图像（已注释）
                        # save_image(frame, event_id)
                        
                    # 处理关节位置输入
                    elif "qpos" in event_id:
                        joints[event_id] = event["value"].to_numpy()
                    # 处理姿态输入
                    elif "pose" in event_id:
                        pose[event_id] = event["value"].to_numpy()
                    
                    # 处理tick事件（触发模型推理）
                    elif "tick" in event_id:
                        logging.info(frames.keys())
                        logging.info(joints.keys())
                        logging.info(pose.keys())
                        # 等待所有图像和数据就绪
                        if len(frames.keys()) < 3:
                            continue
                        if len(joints.keys()) < 2:
                            continue
                        if len(pose.keys()) < 2:
                            continue
                            
                        # 获取机器人关节和姿态数据
                        right_arm_joint = joints["/observations/qpos_right"]
                        left_arm_joint = joints["/observations/qpos_left"]
                        right_arm_pose = pose["/observations/pose_right"][:-1]
                        left_arm_pose = pose["/observations/pose_left"][:-1]
                        logging.info(f" 左臂关节: {left_arm_joint}")
                        logging.info(f" 右臂关节: {right_arm_joint}")
                        logging.info(f" 左臂姿态: {left_arm_pose}")
                        logging.info(f" 右臂姿态: {right_arm_pose}")
                        
                        # 准备观察数据
                        obs = {
                            "observation.state": torch.from_numpy(np.concatenate(
                            [
                                left_arm_joint,
                                right_arm_joint,
                                left_arm_pose,
                                right_arm_pose,
                                np.zeros(12),  # 为7自由度调整零填充维度从12到14
                            ],
                            ).squeeze()).float().unsqueeze(0).to('cuda'),
                            "observation.images.cam_high": TF.resize(torch.from_numpy(np.array(frames['/observations/images/cam_high']).transpose(2, 0, 1)), size=[480, 640]).unsqueeze(0).to('cuda') / 255.0,
                            "observation.images.cam_left_wrist": torch.from_numpy(np.array(frames['/observations/images/cam_left_wrist']).transpose(2, 0, 1)).unsqueeze(0).to('cuda') / 255.0,
                            "observation.images.cam_right_wrist": torch.from_numpy(np.array(frames['/observations/images/cam_right_wrist']).transpose(2, 0, 1)).unsqueeze(0).to('cuda') / 255.0,
                            # "prompt": "stack the brown basket on the black basket",
                            "repo_id": "HuaihaiLyu/rm_groceries"
                        }
                        # 真实动作文件路径（未使用）
                        file_path = '/home/rm/lerobot/outputs/train/realman_groceries_20250501_152526/save_action/gt_action0.npy'
                        
                        # logging.info(f"观察状态_0: {obs['observation.state']}")
                        
                        # 创建保存图像的路径
                        save_dir = os.path.expanduser("~/.images")
                        os.makedirs(save_dir, exist_ok=True)

                        # 图像键与对应的张量
                        image_keys = {
                            "cam_high": torch.from_numpy(np.array(frames['/observations/images/cam_high']).transpose(2, 0, 1)),
                            "cam_left_wrist": torch.from_numpy(np.array(frames['/observations/images/cam_left_wrist']).transpose(2, 0, 1)),
                            "cam_right_wrist": torch.from_numpy(np.array(frames['/observations/images/cam_right_wrist']).transpose(2, 0, 1)),
                        }

                        # 记录观察状态和图像
                        # logging.info(f"观察状态:{obs['observation.state']}")
                        # logging.info(f"高位摄像机图像:{obs['observation.images.cam_high']}")
                        
                        # 使用策略模型选择动作
                        action = policy.select_action(obs).squeeze(0)
                        logging.info(f"--------------action--{action.shape}----------")
                        # logging.info(f"动作_xyz:{action[:10,14:26]}")
                        # 将动作从GPU转移到CPU并转换为numpy数组print("---------------------------", action.shape, "--------------------------")
                        action = action.detach().float().to("cpu").numpy()
                        # 执行生成的动作序列
                        for i in range(action[:25, :].shape[0]):
                            # 分离左右手臂动作，适配7自由度机械臂
                            left_action = action[i, :8]   # 7自由度左臂动作
                            right_action = action[i, 8:16]    # 7自由度右臂动作
                                                   
                            for key, tensor in image_keys.items():
                                # 反归一化和转换为PIL图像
                                image_tensor = tensor.squeeze(0).cpu()  # 去除batch维度
                                image = TF.to_pil_image(image_tensor)

                                # 保存图像
                                save_path = os.path.join(save_dir, f"cam_{key}_{i}.png")
                                image.save(save_path)
                                logging.info(f"保存 {key} 图像, save_path : {save_path}")
                                

                            # print(f"已保存: {save_path}")
                            logging.info(f"左臂动作:{left_action}")
                            logging.info(f"右臂动作:{right_action}")
                            # 发送关节状态到输出
                            node.send_output("jointstate_left", pa.array(left_action.ravel()))
                            node.send_output("jointstate_right", pa.array(right_action.ravel()))
                            time.sleep(0.5)
                        
                        # left_action = action[:8]   # 7自由度左臂动作
                        # right_action = action[8:16]    # 7自由度右臂动作
                        # logging.info(f"左臂动作:{left_action}")
                        # logging.info(f"右臂动作:{right_action}")
                        # # 发送关节状态到输出
                        # node.send_output("jointstate_left", pa.array(left_action.ravel()))
                        # node.send_output("jointstate_right", pa.array(right_action.ravel()))
                        # time.sleep(0.05)
                        time.sleep(1)
if __name__ == "__main__":
    main()

# 命令行运行方式:
# python scirpts/act_infer.py --dataset.repo_id=data/mixed_groceries_bag --policy.type="diffusion"
# python scirpts/act_infer.py --dataset.repo_id=data/mixed_groceries_bag --policy.type="act"