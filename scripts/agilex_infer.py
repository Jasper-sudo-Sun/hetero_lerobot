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
#!/usr/bin/env python3
import rospy
from std_msgs.msg import Header
from data_msgs.msg import Gripper

import os
import time

import numpy as np
import pyarrow as pa
from piper_sdk import C_PiperInterface
import logging

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


def enable_fun(piper: C_PiperInterface):
    """使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序."""
    enable_flag = False
    # 设置超时时间（秒）
    timeout = 5
    # 记录进入循环前的时间
    start_time = time.time()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = (
            piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        )
        print("使能状态:", enable_flag)
        print("--------------------")
        # 检查是否超过超时时间
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
    if elapsed_time_flag:
        print("程序自动使能超时,退出程序")
        raise ConnectionError("程序自动使能超时,退出程序")

class DualPikaGripper:
    def __init__(self):
        rospy.init_node('dual_gripper_commander', anonymous=True)
        self.left_pub = rospy.Publisher('/gripper_l/ctrl', Gripper, queue_size=10)
        self.right_pub = rospy.Publisher('/gripper_r/ctrl', Gripper, queue_size=10)
        rospy.sleep(1.0)  # 等待连接

    def _build_msg(self, angle):
        msg = Gripper()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.enable = True
        msg.setZero = False
        msg.angle = angle
        msg.distance = 0.0
        msg.effort = 0.0
        msg.velocity = 0.0
        return msg

    def enable_left_gripper(self):
        msg = self._build_msg(0.0)
        self.left_pub.publish(msg)
        rospy.loginfo("Left gripper enabled.")

    def enable_right_gripper(self):
        msg = self._build_msg(0.0)
        self.right_pub.publish(msg)
        rospy.loginfo("Right gripper enabled.")

    def set_left_angle(self, angle):
        msg = self._build_msg(angle)
        self.left_pub.publish(msg)
        rospy.loginfo(f"Left gripper set to {angle} rad.")

    def set_right_angle(self, angle):
        msg = self._build_msg(angle)
        self.right_pub.publish(msg)
        rospy.loginfo(f"Right gripper set to {angle} rad.")

    def enable_both(self):
        msg = self._build_msg(0.0)
        self.left_pub.publish(msg)
        self.right_pub.publish(msg)
        rospy.loginfo("Both grippers enabled.")

    def set_both_angles(self, left_angle, right_angle):
        left_msg = self._build_msg(left_angle)
        right_msg = self._build_msg(right_angle)
        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)
        rospy.loginfo(f"Left gripper: {left_angle} rad, Right gripper: {right_angle} rad.")

import rospy
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray, Header
from data_msgs.msg import Gripper
import os
import time
import numpy as np
from piper_sdk import C_PiperInterface
def get_obs():
    # TODO: LiMing Chen
    """
    获取所有观测数据的函数，包括机械臂状态、相机图像和夹爪状态
    Returns:
        dict: 包含所有观测数据的字典
    """
    try:
        # 初始化ROS节点
        if not rospy.is_initialized():
            rospy.init_node('obs_collector', anonymous=True)
        
        print("🚀 开始初始化双臂系统...")
        
        # 初始化数据存储变量
        obs_data = {
            'fisheye_l_image': None,
            'fisheye_r_image': None,
            'realsense_l_color_image': None,
            'realsense_r_color_image': None,
            'gripper_l_state': None,
            'gripper_r_state': None,
            'left_arm_position': None,
            'right_arm_position': None
        }
        
        # 定义回调函数
        def fisheye_l_image_callback(msg):
            obs_data['fisheye_l_image'] = msg
            
        def fisheye_r_image_callback(msg):
            obs_data['fisheye_r_image'] = msg
            
        def realsense_l_color_image_callback(msg):
            obs_data['realsense_l_color_image'] = msg
            
        def realsense_r_color_image_callback(msg):
            obs_data['realsense_r_color_image'] = msg
            
        def gripper_l_state_callback(msg):
            obs_data['gripper_l_state'] = msg
            
        def gripper_r_state_callback(msg):
            obs_data['gripper_r_state'] = msg

        # 创建订阅者
        rospy.Subscriber("/camera_fisheye_l/color/image_raw", Image, fisheye_l_image_callback)
        rospy.Subscriber("/camera_fisheye_r/color/image_raw", Image, fisheye_r_image_callback)
        rospy.Subscriber("/camera_l/color/image_raw", Image, realsense_l_color_image_callback)
        rospy.Subscriber("/camera_r/color/image_raw", Image, realsense_r_color_image_callback)
        rospy.Subscriber("/gripper_l/joint_states", JointState, gripper_l_state_callback)
        rospy.Subscriber("/gripper_r/joint_states", JointState, gripper_r_state_callback)

        # 创建夹爪控制器发布者
        left_gripper_pub = rospy.Publisher('/gripper_l/ctrl', Gripper, queue_size=10)
        right_gripper_pub = rospy.Publisher('/gripper_r/ctrl', Gripper, queue_size=10)

        # 使能夹爪
        gripper_msg = Gripper()
        gripper_msg.header = Header()
        gripper_msg.header.stamp = rospy.Time.now()
        gripper_msg.enable = True
        gripper_msg.setZero = False
        gripper_msg.angle = 0.0
        gripper_msg.distance = 0.0
        gripper_msg.effort = 0.0
        gripper_msg.velocity = 0.0

        left_gripper_pub.publish(gripper_msg)
        right_gripper_pub.publish(gripper_msg)
        print("✅ 夹爪控制器初始化完成")

        # 初始化机械臂
        print("🔧 正在连接左机械臂...")
        piper_left = C_PiperInterface("can_left", False)
        piper_left.ConnectPort()
        piper_left.EnableArm(7)
        enable_fun(piper=piper_left)
        
        piper_left.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        piper_left.JointCtrl(0, 0, 0, 0, 0, 0)
        piper_left.GripperCtrl(abs(0), 1000, 0x01, 0)
        piper_left.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        print("✅ 左机械臂初始化完成")

        print("🔧 正在连接右机械臂...")
        piper_right = C_PiperInterface("can_right", False)
        piper_right.ConnectPort()
        piper_right.EnableArm(7)
        enable_fun(piper=piper_right)
        
        piper_right.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        piper_right.JointCtrl(0, 0, 0, 0, 0, 0)
        piper_right.GripperCtrl(abs(0), 1000, 0x01, 0)
        piper_right.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        print("✅ 右机械臂初始化完成")

        # 等待所有数据就绪
        print("📊 等待数据收集...")
        timeout = time.time() + 10.0  # 10秒超时
        while not rospy.is_shutdown():
            if all(value is not None for value in obs_data.values()):
                break
            if time.time() > timeout:
                raise TimeoutError("等待数据超时")
            rospy.sleep(0.1)

        # 获取机械臂位置
        def get_piper_position(piper, arm_name):
            try:
                position = piper.GetArmEndPoseMsgs()
                position_value = []
                position_value += [position.end_pose.X_axis * 0.001 * 0.001]
                position_value += [position.end_pose.Y_axis * 0.001 * 0.001]
                position_value += [position.end_pose.Z_axis * 0.001 * 0.001]
                position_value += [position.end_pose.RX_axis * 0.001 / 360 * 2 * np.pi]
                position_value += [position.end_pose.RY_axis * 0.001 / 360 * 2 * np.pi]
                position_value += [position.end_pose.RZ_axis * 0.001 / 360 * 2 * np.pi]
                
                print(f"\n========== {arm_name} Piper 机械臂实时位置 ==========")
                print(f"位置 (m): X={position_value[0]:.6f}, Y={position_value[1]:.6f}, Z={position_value[2]:.6f}")
                print(f"姿态 (rad): RX={position_value[3]:.6f}, RY={position_value[4]:.6f}, RZ={position_value[5]:.6f}")
                print(f"原始数据: X={position.end_pose.X_axis}, Y={position.end_pose.Y_axis}, Z={position.end_pose.Z_axis}")
                print(f"原始姿态: RX={position.end_pose.RX_axis}, RY={position.end_pose.RY_axis}, RZ={position.end_pose.RZ_axis}")
                print("=" * 50)
                
                return position_value
            except Exception as e:
                print(f"❌ 获取{arm_name}机械臂位置失败: {e}")
                return [0.0] * 6

        # 获取机械臂位置数据
        print(f"\n🤖 [{time.strftime('%H:%M:%S')}] 正在获取机械臂位置信息...")
        obs_data['left_arm_position'] = get_piper_position(piper_left, "Left")
        obs_data['right_arm_position'] = get_piper_position(piper_right, "Right")

        # 打印数据状态
        print(f"\n📷 [{time.strftime('%H:%M:%S')}] 相机数据状态:")
        print(f"  左鱼眼相机: {obs_data['fisheye_l_image'].width}x{obs_data['fisheye_l_image'].height}")
        print(f"  右鱼眼相机: {obs_data['fisheye_r_image'].width}x{obs_data['fisheye_r_image'].height}")
        print(f"  左RealSense: {obs_data['realsense_l_color_image'].width}x{obs_data['realsense_l_color_image'].height}")
        print(f"  右RealSense: {obs_data['realsense_r_color_image'].width}x{obs_data['realsense_r_color_image'].height}")

        print(f"\n🦾 [{time.strftime('%H:%M:%S')}] 夹爪状态:")
        print(f"  左夹爪位置: {obs_data['gripper_l_state'].position}")
        print(f"  右夹爪位置: {obs_data['gripper_r_state'].position}")

        print(f"\n✅ [{time.strftime('%H:%M:%S')}] 观测数据收集完成")
        print(f"{'='*60}")

        return obs_data

    except Exception as e:
        print(f"❌ 获取观测数据失败: {e}")
        return None


def action_publish(pred_action):
    """
        input: pred_action: [chunk_size, 38]
        output: None
    """
    
    #左右两pika夹爪控制
    action_left = pred_action[0, 0:7]
    eef_left = action_left[0:6]
    gripper_left = action_left[6]
    action_right = pred_action[0, 7:14]
    eef_right = action_right[0:6]
    gripper_right = action_right[6]

    gripper = DualPikaGripper()
    gripper.enable_left_gripper()
    gripper.enable_right_gripper()
    gripper.set_both_angles(gripper_left, gripper_right)

    #---------------左 piper机械臂-------------------------
    piper_left = C_PiperInterface("can_left", False)
    piper_left.ConnectPort()
    piper_left.EnableArm(7)
    enable_fun(piper=piper_left)
    piper_left.MotionCtrl_2(0x01, 0x01, 50, 0x00)
    piper_left.JointCtrl(0, 0, 0, 0, 0, 0)
    piper_left.GripperCtrl(abs(0), 1000, 0x01, 0)
    piper_left.MotionCtrl_2(0x01, 0x01, 50, 0x00)
    time.sleep(5)

    X = round(eef_left[0] * 1000 * 1000)
    Y = round(eef_left[1] * 1000 * 1000)
    Z = round(eef_left[2] * 1000 * 1000)
    RX = round(eef_left[3] * 1000 / (2 * np.pi) * 360)
    RY = round(eef_left[4] * 1000 / (2 * np.pi) * 360)
    RZ = round(eef_left[5] * 1000 / (2 * np.pi) * 360)
    piper_left.EndPoseCtrl(
        X ,
        Y,
        Z,
        RX,
        RY,
        RZ,
    )
    piper_left.MotionCtrl_2(0x01, 0x00, 50, 0x00)
    time.sleep(1)

    #---------------右piper机械臂-------------------------
    piper_right = C_PiperInterface("can_right", False)
    piper_right.ConnectPort()
    piper_right.EnableArm(7)
    enable_fun(piper=piper_right)
    piper_right.MotionCtrl_2(0x01, 0x01, 50, 0x00)
    piper_right.JointCtrl(0, 0, 0, 0, 0, 0)
    piper_right.GripperCtrl(abs(0), 1000, 0x01, 0)
    piper_right.MotionCtrl_2(0x01, 0x01, 50, 0x00)

    X = round(eef_right[0] * 1000 * 1000)
    Y = round(eef_right[1] * 1000 * 1000)
    Z = round(eef_right[2] * 1000 * 1000)
    RX = round(eef_right[3] * 1000 / (2 * np.pi) * 360)
    RY = round(eef_right[4] * 1000 / (2 * np.pi) * 360)
    RZ = round(eef_right[5] * 1000 / (2 * np.pi) * 360)
    piper_left.EndPoseCtrl(
        X ,
        Y,
        Z,
        RX,
        RY,
        RZ,
    )
    piper_right.MotionCtrl_2(0x01, 0x00, 50, 0x00)

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
            # === obs acquire
            observation, abs_pose = get_obs()
            # === infer
            pred_action = policy.forward_action(observation).squeeze()
            # === 将当前相对动作转换为绝对位姿序列
            pred_abs_action = relative_to_absolute_poses(pred_action.cpu().numpy(), abs_pose)
            
            for i in range(50):
                action_publish(pred_abs_action[i])

            import pdb
            pdb.set_trace()

    