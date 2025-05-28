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

if __name__ == '__main__':
    #左右两pika夹爪控制
    gripper = DualPikaGripper()

    gripper.enable_left_gripper()
    rospy.sleep(1)
    gripper.set_left_angle(1.0)
    rospy.sleep(1)

    gripper.enable_right_gripper()
    rospy.sleep(3)
    gripper.set_right_angle(1.0)
    rospy.sleep(3)
    # 同时控制两个pika夹爪闭合
    # gripper.set_both_angles(0.0, 0.0)

    """TODO: Add docstring."""
    # CAN_BUS = os.getenv("CAN_BUS", "can_right")
    time.sleep(5)
    
    #左piper机械臂
    piper_left = C_PiperInterface("can_left", False)
    piper_left.ConnectPort()
    # import pdb
    # pdb.set_trace()
    piper_left.EnableArm(7)
    enable_fun(piper=piper_left)
    piper_left.MotionCtrl_2(0x01, 0x01, 50, 0x00)
    piper_left.JointCtrl(0, 0, 0, 0, 0, 0)
    piper_left.GripperCtrl(abs(0), 1000, 0x01, 0)
    piper_left.MotionCtrl_2(0x01, 0x01, 50, 0x00)
    time.sleep(5)
    
    #右piper机械臂    
    piper_right = C_PiperInterface("can_right", False)
    piper_right.ConnectPort()
    # import pdb
    # pdb.set_trace()
    piper_right.EnableArm(7)
    enable_fun(piper=piper_right)
    piper_right.MotionCtrl_2(0x01, 0x01, 50, 0x00)
    piper_right.JointCtrl(0, 0, 0, 0, 0, 0)
    piper_right.GripperCtrl(abs(0), 1000, 0x01, 0)
    piper_right.MotionCtrl_2(0x01, 0x01, 50, 0x00)
    time.sleep(5)


    # factor = 57324.840764  # 1000*180/3.14
    # piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)

    # X = round(0 * 1000 * 1000)
    # Y = round(0 * 1000 * 1000)
    # Z = round(0 * 1000 * 1000)
    # RX = round(0 * 1000 / (2 * np.pi) * 360)
    # RY = round(0 * 1000 / (2 * np.pi) * 360)
    # RZ = round(0 * 1000 / (2 * np.pi) * 360)
    # piper.EndPoseCtrl(
    #     X ,
    #     Y,
    #     Z,
    #     RX,
    #     RY,
    #     RZ,
    # )
    # piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)
