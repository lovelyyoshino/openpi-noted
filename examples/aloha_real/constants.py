# 忽略lint错误，因为这个文件主要是从ACT（https://github.com/tonyzhaozh/act）复制而来。
# ruff: noqa

### 任务参数

### ALOHA固定常量
DT = 0.001  # 时间步长，单位：秒
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]  # 机械臂关节名称列表
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239, 0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]  # 初始机械臂姿势的关节角度配置

# 左手指位置限制 (qpos[7]), 右食指位置与左食指相反
MASTER_GRIPPER_POSITION_OPEN = 0.02417  # 主夹爪打开时的位置
MASTER_GRIPPER_POSITION_CLOSE = 0.01244  # 主夹爪关闭时的位置
PUPPET_GRIPPER_POSITION_OPEN = 0.05800  # 木偶夹爪打开时的位置
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844  # 木偶夹爪关闭时的位置

# 夹具关节限制 (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083  # 主夹爪打开时的关节角度
MASTER_GRIPPER_JOINT_CLOSE = -0.6842  # 主夹爪关闭时的关节角度
PUPPET_GRIPPER_JOINT_OPEN = 1.4910  # 木偶夹爪打开时的关节角度
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213  # 木偶夹爪关闭时的关节角度

############################ 辅助函数 ############################

# 主夹爪位置归一化函数，将位置映射到[0, 1]范围
MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (
    MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE
)

# 木偶夹爪位置归一化函数，将位置映射到[0, 1]范围
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (
    PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
)

# 主夹爪位置反归一化函数，从[0, 1]转换回实际位置
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
)

# 木偶夹爪位置反归一化函数，从[0, 1]转换回实际位置
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
)

# 将主夹爪位置转换为木偶夹爪位置的混合函数
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

# 主夹爪关节角度归一化函数，将角度映射到[0, 1]范围
MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (
    MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE
)

# 木偶夹爪关节角度归一化函数，将角度映射到[0, 1]范围
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (
    PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE
)

# 主夹爪关节角度反归一化函数，从[0, 1]转换回实际角度
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = (
    lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
)

# 木偶夹爪关节角度反归一化函数，从[0, 1]转换回实际角度
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = (
    lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
)

# 将主夹爪关节角度转换为木偶夹爪关节角度的混合函数
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

# 主夹爪速度归一化函数，将速度映射到[-1, 1]范围
MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)

# 木偶夹爪速度归一化函数，将速度映射到[-1, 1]范围
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

# 主夹爪位置到关节角度的转换函数
MASTER_POS2JOINT = (
    lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    + MASTER_GRIPPER_JOINT_CLOSE
)

# 主夹爪关节角度到位置的转换函数
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
)

# 木偶夹爪位置到关节角度的转换函数
PUPPET_POS2JOINT = (
    lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    + PUPPET_GRIPPER_JOINT_CLOSE
)

# 木偶夹爪关节角度到位置的转换函数
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
)

# 主夹爪关节中间值，用于定义中间状态
MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE) / 2  
