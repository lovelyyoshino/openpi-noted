# 忽略lint错误，因为该文件大多数内容来自ACT (https://github.com/tonyzhaozh/act)。
# ruff: noqa
import collections
import time
from typing import Optional, List
import dm_env 
from interbotix_xs_modules.arm import InterbotixManipulatorXS  # 导入Interbotix机器人操作类
from interbotix_xs_msgs.msg import JointSingleCommand  # 导入关节单一命令消息类型
import numpy as np

from examples.aloha_real import constants 
from examples.aloha_real import robot_utils

# 这是标准Aloha运行时使用的重置位置
DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]  # 默认的机器臂重置姿势


class RealEnv:
    """
    用于真实机器人双手操作的环境
    动作空间:      [左臂关节位置 (6),             # 绝对关节位置
                    左夹爪位置 (1),               # 归一化夹爪位置 (0: 关闭, 1: 打开)
                    右臂关节位置 (6),            # 绝对关节位置
                    右夹爪位置 (1)]               # 归一化夹爪位置 (0: 关闭, 1: 打开)

    观测空间: {"qpos": 联合[左臂关节位置 (6),           # 绝对关节位置
                              左夹爪位置 (1),         # 归一化夹爪位置 (0: 关闭, 1: 打开)
                              右臂关节位置 (6),       # 绝对关节位置
                              右夹爪关节位置 (1)],   # 归一化夹爪位置 (0: 关闭, 1: 打开)
                "qvel": 联合[左臂关节速度 (6),        # 绝对关节速度 (弧度)
                              左夹爪速度 (1),          # 归一化夹爪速度（正：打开，负：关闭）
                              右臂关节速度 (6),       # 绝对关节速度 (弧度)
                              右夹爪关节速度 (1)]     # 归一化夹爪速度（正：打开，负：关闭）
                "images": {"cam_high": (480x640x3),   # 高分辨率相机图像，维度为 h, w, c, dtype='uint8'
                           "cam_low": (480x640x3),    # 低分辨率相机图像
                           "cam_left_wrist": (480x640x3),  # 左腕相机图像
                           "cam_right_wrist": (480x640x3)} # 右腕相机图像
    """

    def __init__(self, init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True):#这个中间*表示后面的参数只能通过关键字传递
        # 初始化参数并设置关节初始位置
        self._reset_position = reset_position[:6] if reset_position else DEFAULT_RESET_POSITION

        # 来自左侧的机器人设置
        self.puppet_bot_left = InterbotixManipulatorXS(
            robot_model="vx300s",
            group_name="arm",
            gripper_name="gripper",
            robot_name="puppet_left",
            init_node=init_node,
        )
        # 来自右侧的机器人设置
        self.puppet_bot_right = InterbotixManipulatorXS(
            robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name="puppet_right", init_node=False
        )
        
        # 如果需要设置机器人，则调用setup_robots方法
        if setup_robots:
            self.setup_robots()

        # 创建记录器以存储状态信息
        self.recorder_left = robot_utils.Recorder("left", init_node=False)
        self.recorder_right = robot_utils.Recorder("right", init_node=False)
        self.image_recorder = robot_utils.ImageRecorder(init_node=False)
        self.gripper_command = JointSingleCommand(name="gripper")  # 创建夹爪指令对象

    def setup_robots(self):
        """设置机器人"""
        robot_utils.setup_puppet_bot(self.puppet_bot_left)
        robot_utils.setup_puppet_bot(self.puppet_bot_right)

    def get_qpos(self):
        """
        获取当前关节位置
        返回：
            numpy.ndarray: 含有左右两只手臂及其抓取器当前关节位置的数组
        """
        left_qpos_raw = self.recorder_left.qpos  # 获取左边机器人的原始关节位置
        right_qpos_raw = self.recorder_right.qpos  # 获取右边机器人的原始关节位置
        
        # 提取左侧臂、右侧臂与夹爪的位置
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [
            constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[7]) 
        ]  # 将左夹爪的位置进行归一化处理
        right_gripper_qpos = [
            constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[7])
        ]  # 将右夹爪的位置进行归一化处理
        
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    def get_qvel(self):
        """
        获取当前关节速度
        返回：
            numpy.ndarray: 含有左右两只手臂及其抓取器当前关节速度的数组
        """
        left_qvel_raw = self.recorder_left.qvel  # 获取左边机器人的原始关节速度
        right_qvel_raw = self.recorder_right.qvel  # 获取右边机器人的原始关节速度
        
        # 提取左侧臂、右侧臂与夹爪的速度
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[7])]  # 规范化
        right_gripper_qvel = [constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[7])]
        
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    def get_effort(self):
        """
        获取当前施力情况
        返回：
            numpy.ndarray: 含有左右两只手臂实际施加的努力指数
        """
        left_effort_raw = self.recorder_left.effort  # 获取左侧机器人的施力数据
        right_effort_raw = self.recorder_right.effort  # 获取右侧机器人的施力数据
        
        left_robot_effort = left_effort_raw[:7]  # 仅选择有效的施力部分
        right_robot_effort = right_effort_raw[:7]  # 仅选择有效的施力部分
        
        return np.concatenate([left_robot_effort, right_robot_effort])

    def get_images(self):
        """获取相机图像"""
        return self.image_recorder.get_images()  # 从图像记录器中获取图像

    def set_gripper_pose(self, left_gripper_desired_pos_normalized, right_gripper_desired_pos_normalized):
        """
        设置夹爪的期望位置
        参数:
            left_gripper_desired_pos_normalized: 左夹爪期望的归一化夹爪位置
            right_gripper_desired_pos_normalized: 右夹爪期望的归一化夹爪位置
        """
        left_gripper_desired_joint = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(left_gripper_desired_pos_normalized)  # 解归一化
        self.gripper_command.cmd = left_gripper_desired_joint  # 更新左夹爪请求
        self.puppet_bot_left.gripper.core.pub_single.publish(self.gripper_command)  # 发布命令

        right_gripper_desired_joint = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
            right_gripper_desired_pos_normalized 
        )  # 解归一化
        self.gripper_command.cmd = right_gripper_desired_joint  # 更新右夹爪请求
        self.puppet_bot_right.gripper.core.pub_single.publish(self.gripper_command)  # 发布命令

    def _reset_joints(self):
        """重置机器人的关节到指定的初始位置"""
        robot_utils.move_arms(
            [self.puppet_bot_left, self.puppet_bot_right], [self._reset_position, self._reset_position], move_time=1
        )

    def _reset_gripper(self):
        """将夹爪设置为位置模式，并执行位置归零：首先开放，然后关闭。然后改回PWM模式"""
        robot_utils.move_grippers(
            [self.puppet_bot_left, self.puppet_bot_right], [constants.PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5
        )  # 首先打开夹爪
        robot_utils.move_grippers(
            [self.puppet_bot_left, self.puppet_bot_right], [constants.PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=1
        )  # 然后关闭夹爪

    def get_observation(self):
        """
        获取当前环境的观察结果
        返回：
            OrderedDict: 环境状态的字典，包括关节位置、速度、施力与图像数据
        """
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()  # 获取关节位置
        obs["qvel"] = self.get_qvel()  # 获取关节速度
        obs["effort"] = self.get_effort()  # 获取施力
        obs["images"] = self.get_images()  # 获取图像
        return obs

    def get_reward(self):
        """获取当前奖励，这里返回0表示未完成任务"""
        return 0

    def reset(self, *, fake=False):
        """
        重置环境
        参数:
            fake: 是否进行虚拟复位，不影响硬件
        返回：
            dm_env.TimeStep: 新的时间步长数据
        """
        if not fake:
            # 重启玩偶机器人夹爪电机
            self.puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
            self.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
            self._reset_joints()  # 重置关节到起始位置
            self._reset_gripper()  # 重置夹爪为初始状态
            
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST, reward=self.get_reward(), discount=None, observation=self.get_observation()  # 返回第一个步骤的信息
        )

    def step(self, action):
        """
        执行一步动作
        参数:
            action: 要执行的动作
        返回：
            dm_env.TimeStep: 后续时间步长信息
        """
        state_len = int(len(action) / 2)  # 每只机器人各占一半动作长度
        left_action = action[:state_len]  # 分离出左侧动作
        right_action = action[state_len:]  # 分离出右侧动作
        
        # 为左、右机器臂设置关节目标位置
        self.puppet_bot_left.arm.set_joint_positions(left_action[:6], blocking=False)
        self.puppet_bot_right.arm.set_joint_positions(right_action[:6], blocking=False)
        
        # 设置夹爪的期望位置
        self.set_gripper_pose(left_action[-1], right_action[-1])
        time.sleep(constants.DT)  # 暂停一段时间，根据常量DT设定
        
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()  # 返回当前步骤的信息
        )


def get_action(master_bot_left, master_bot_right):
    """
    从主机器人获取当前动作
    参数:
        master_bot_left: 左侧主机器人实例
        master_bot_right: 右侧主机器人实例
    返回：
        numpy.ndarray: 当前动作向量
    """
    action = np.zeros(14)  # 包含6个关节 + 1个夹爪，共两个手臂
    # 读取臂部动作
    action[:6] = master_bot_left.dxl.joint_states.position[:6]  # 左侧六个关节
    action[7 : 7 + 6] = master_bot_right.dxl.joint_states.position[:6]  # 右侧六个关节
    
    # 读取夹爪动作
    action[6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])  # 左夹爪
    action[7 + 6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])  # 右夹爪

    return action  # 返回动作向量


def make_real_env(init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True) -> RealEnv:
    """创建真实环境实例
    参数:
        init_node: 初始化节点
        reset_position: 可选的重置位置
        setup_robots: 布署机器人标志
    返回:
        RealEnv: 真实环境实例
    """
    return RealEnv(init_node, reset_position=reset_position, setup_robots=setup_robots)
