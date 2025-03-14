# 忽略lint错误，因为这个文件大部分是从ACT（https://github.com/tonyzhaozh/act）复制的。
# ruff: noqa
from collections import deque
import datetime
import json
import time

from aloha.msg import RGBGrayscaleImage
from cv_bridge import CvBridge
from interbotix_xs_msgs.msg import JointGroupCommand
from interbotix_xs_msgs.msg import JointSingleCommand
import numpy as np
import rospy
from sensor_msgs.msg import JointState

from examples.aloha_real import constants


class ImageRecorder:
    def __init__(self, init_node=True, is_debug=False):
        # 初始化图像记录器，设置调试模式和相机名称
        self.is_debug = is_debug
        self.bridge = CvBridge()  # 创建CvBridge对象用于图像转换
        self.camera_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

        if init_node:
            rospy.init_node("image_recorder", anonymous=True)  # 初始化ROS节点
        for cam_name in self.camera_names:
            setattr(self, f"{cam_name}_rgb_image", None)  # 设置RGB图像属性
            setattr(self, f"{cam_name}_depth_image", None)  # 设置深度图像属性
            setattr(self, f"{cam_name}_timestamp", 0.0)  # 设置时间戳属性
            # 根据相机名称选择回调函数
            if cam_name == "cam_high":
                callback_func = self.image_cb_cam_high
            elif cam_name == "cam_low":
                callback_func = self.image_cb_cam_low
            elif cam_name == "cam_left_wrist":
                callback_func = self.image_cb_cam_left_wrist
            elif cam_name == "cam_right_wrist":
                callback_func = self.image_cb_cam_right_wrist
            else:
                raise NotImplementedError
            rospy.Subscriber(f"/{cam_name}", RGBGrayscaleImage, callback_func)  # 订阅相机话题
            if self.is_debug:
                setattr(self, f"{cam_name}_timestamps", deque(maxlen=50))  # 存储时间戳以进行调试

        self.cam_last_timestamps = {cam_name: 0.0 for cam_name in self.camera_names}  # 初始化最后时间戳字典
        time.sleep(0.5)  # 等待一段时间以确保所有订阅者都已连接

    def image_cb(self, cam_name, data):
        # 图像回调函数，用于处理接收到的图像数据
        setattr(
            self,
            f"{cam_name}_rgb_image",
            self.bridge.imgmsg_to_cv2(data.images[0], desired_encoding="bgr8"),  # 转换为OpenCV格式
        )
        # setattr(
        #     self,
        #     f"{cam_name}_depth_image",
        #     self.bridge.imgmsg_to_cv2(data.images[1], desired_encoding="mono16"),
        # )
        setattr(
            self,
            f"{cam_name}_timestamp",
            data.header.stamp.secs + data.header.stamp.nsecs * 1e-9,  # 获取时间戳
        )
        # setattr(self, f'{cam_name}_secs', data.images[0].header.stamp.secs)
        # setattr(self, f'{cam_name}_nsecs', data.images[0].header.stamp.nsecs)
        # cv2.imwrite('/home/lucyshi/Desktop/sample.jpg', cv_image)
        if self.is_debug:
            getattr(self, f"{cam_name}_timestamps").append(
                data.images[0].header.stamp.secs + data.images[0].header.stamp.nsecs * 1e-9
            )  # 在调试模式下存储时间戳

    def image_cb_cam_high(self, data):
        # 高位相机的图像回调
        cam_name = "cam_high"
        return self.image_cb(cam_name, data)

    def image_cb_cam_low(self, data):
        # 低位相机的图像回调
        cam_name = "cam_low"
        return self.image_cb(cam_name, data)

    def image_cb_cam_left_wrist(self, data):
        # 左腕相机的图像回调
        cam_name = "cam_left_wrist"
        return self.image_cb(cam_name, data)

    def image_cb_cam_right_wrist(self, data):
        # 右腕相机的图像回调
        cam_name = "cam_right_wrist"
        return self.image_cb(cam_name, data)

    def get_images(self):
        # 获取所有相机的图像
        image_dict = {}
        for cam_name in self.camera_names:
            while getattr(self, f"{cam_name}_timestamp") <= self.cam_last_timestamps[cam_name]:
                time.sleep(0.00001)  # 等待直到新的时间戳可用
            rgb_image = getattr(self, f"{cam_name}_rgb_image")
            depth_image = getattr(self, f"{cam_name}_depth_image")
            self.cam_last_timestamps[cam_name] = getattr(self, f"{cam_name}_timestamp")  # 更新最后时间戳
            image_dict[cam_name] = rgb_image  # 将RGB图像添加到字典中
            image_dict[f"{cam_name}_depth"] = depth_image  # 将深度图像添加到字典中
        return image_dict  # 返回包含所有图像的字典

    def print_diagnostics(self):
        # 打印诊断信息
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)  # 计算时间差的平均值

        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f"{cam_name}_timestamps"))  # 计算图像频率
            print(f"{cam_name} {image_freq=:.2f}")  # 打印每个相机的频率
        print()


class Recorder:
    def __init__(self, side, init_node=True, is_debug=False):
        # 初始化记录器，设置侧面、调试模式等
        self.secs = None
        self.nsecs = None
        self.qpos = None
        self.effort = None
        self.arm_command = None
        self.gripper_command = None
        self.is_debug = is_debug

        if init_node:
            rospy.init_node("recorder", anonymous=True)  # 初始化ROS节点
        rospy.Subscriber(f"/puppet_{side}/joint_states", JointState, self.puppet_state_cb)  # 订阅关节状态
        rospy.Subscriber(
            f"/puppet_{side}/commands/joint_group",
            JointGroupCommand,
            self.puppet_arm_commands_cb,  # 订阅手臂命令
        )
        rospy.Subscriber(
            f"/puppet_{side}/commands/joint_single",
            JointSingleCommand,
            self.puppet_gripper_commands_cb,  # 订阅夹爪命令
        )
        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)  # 存储关节时间戳以进行调试
            self.arm_command_timestamps = deque(maxlen=50)  # 存储手臂命令时间戳
            self.gripper_command_timestamps = deque(maxlen=50)  # 存储夹爪命令时间戳
        time.sleep(0.1)  # 等待一段时间以确保所有订阅者都已连接

    def puppet_state_cb(self, data):
        # 玩偶状态回调函数
        self.qpos = data.position  # 获取位置
        self.qvel = data.velocity  # 获取速度
        self.effort = data.effort  # 获取努力值
        self.data = data  # 保存原始数据
        if self.is_debug:
            self.joint_timestamps.append(time.time())  # 在调试模式下存储时间戳

    def puppet_arm_commands_cb(self, data):
        # 手臂命令回调函数
        self.arm_command = data.cmd  # 获取手臂命令
        if self.is_debug:
            self.arm_command_timestamps.append(time.time())  # 在调试模式下存储时间戳

    def puppet_gripper_commands_cb(self, data):
        # 夹爪命令回调函数
        self.gripper_command = data.cmd  # 获取夹爪命令
        if self.is_debug:
            self.gripper_command_timestamps.append(time.time())  # 在调试模式下存储时间戳

    def print_diagnostics(self):
        # 打印诊断信息
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)  # 计算时间差的平均值

        joint_freq = 1 / dt_helper(self.joint_timestamps)  # 计算关节频率
        arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)  # 计算手臂命令频率
        gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)  # 计算夹爪命令频率

        print(f"{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n{gripper_command_freq=:.2f}\n")  # 打印频率


def get_arm_joint_positions(bot):
    # 获取机器人手臂的关节位置
    return bot.arm.core.joint_states.position[:6]


def get_arm_gripper_positions(bot):
    # 获取机器人夹爪的位置
    return bot.gripper.core.joint_states.position[6]


def move_arms(bot_list, target_pose_list, move_time=1):
    # 移动多个机器人的手臂到目标位置
    num_steps = int(move_time / constants.DT)  # 计算移动步骤数
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]  # 获取当前姿态列表
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps)  # 生成轨迹
        for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
    ]
    for t in range(num_steps):  # 遍历每一步
        for bot_id, bot in enumerate(bot_list):
            bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)  # 设置手臂关节位置
        time.sleep(constants.DT)  # 等待下一步

def move_grippers(bot_list, target_pose_list, move_time):
    # 移动多个机器人的夹爪到目标位置
    print(f"Moving grippers to {target_pose_list=}")
    gripper_command = JointSingleCommand(name="gripper")  # 创建夹爪命令对象
    num_steps = int(move_time / constants.DT)  # 计算移动步骤数
    curr_pose_list = [get_arm_gripper_positions(bot) for bot in bot_list]  # 获取当前夹爪位置
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps)  # 生成轨迹
        for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
    ]

    with open(f"/data/gripper_traj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl", "a") as f:
        for t in range(num_steps):  # 遍历每一步
            d = {}
            for bot_id, bot in enumerate(bot_list):
                gripper_command.cmd = traj_list[bot_id][t]  # 设置夹爪命令
                bot.gripper.core.pub_single.publish(gripper_command)  # 发布夹爪命令
                d[bot_id] = {"obs": get_arm_gripper_positions(bot), "act": traj_list[bot_id][t]}  # 记录观察和动作
            f.write(json.dumps(d) + "\n")  # 写入日志文件
            time.sleep(constants.DT)  # 等待下一步


def setup_puppet_bot(bot):
    # 设置玩偶机器人
    bot.dxl.robot_reboot_motors("single", "gripper", True)  # 重启夹爪电机
    bot.dxl.robot_set_operating_modes("group", "arm", "position")  # 设置手臂操作模式为位置控制
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")  # 设置夹爪操作模式为基于电流的位置控制
    torque_on(bot)  # 开启扭矩


def setup_master_bot(bot):
    # 设置主控机器人
    bot.dxl.robot_set_operating_modes("group", "arm", "pwm")  # 设置手臂操作模式为PWM控制
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")  # 设置夹爪操作模式为基于电流的位置控制
    torque_off(bot)  # 关闭扭矩


def set_standard_pid_gains(bot):
    # 设置标准PID增益
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_P_Gain", 800)  # 设置P增益
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_I_Gain", 0)  # 设置I增益


def set_low_pid_gains(bot):
    # 设置低PID增益
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_P_Gain", 100)  # 设置P增益
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_I_Gain", 0)  # 设置I增益


def torque_off(bot):
    # 关闭扭矩
    bot.dxl.robot_torque_enable("group", "arm", False)  # 关闭手臂扭矩
    bot.dxl.robot_torque_enable("single", "gripper", False)  # 关闭夹爪扭矩


def torque_on(bot):
    # 开启扭矩
    bot.dxl.robot_torque_enable("group", "arm", True)  # 开启手臂扭矩
    bot.dxl.robot_torque_enable("single", "gripper", True)  # 开启夹爪扭矩


# 用于DAgger同步
def sync_puppet_to_master(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right):
    # 同步玩偶与主控机器人
    print("\nSyncing!")

    # 激活主控机器人的手臂
    torque_on(master_bot_left)
    torque_on(master_bot_right)

    # 获取玩偶机器人的手臂位置
    puppet_left_qpos = get_arm_joint_positions(puppet_bot_left)
    puppet_right_qpos = get_arm_joint_positions(puppet_bot_right)

    # 获取玩偶机器人的夹爪位置
    puppet_left_gripper = get_arm_gripper_positions(puppet_bot_left)
    puppet_right_gripper = get_arm_gripper_positions(puppet_bot_right)

    # 将主控机器人的手臂移动到玩偶机器人的位置
    move_arms(
        [master_bot_left, master_bot_right],
        [puppet_left_qpos, puppet_right_qpos],
        move_time=1,
    )

    # 将主控机器人的夹爪移动到玩偶机器人的位置
    move_grippers(
        [master_bot_left, master_bot_right],
        [puppet_left_gripper, puppet_right_gripper],
        move_time=1,
    )
