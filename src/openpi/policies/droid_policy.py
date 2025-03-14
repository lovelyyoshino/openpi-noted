import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_droid_example() -> dict:
    """创建一个Droid策略的随机输入示例。"""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # 随机生成左侧外部图像，形状为(224, 224, 3)，值范围在0到255之间。
        
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # 随机生成左腕图像，同样形状和取值。
        
        "observation/joint_position": np.random.rand(7),
        # 随机生成关节位置，包含7个位置坐标。
        
        "observation/gripper_position": np.random.rand(1),
        # 随机生成抓手位置，仅有一个维度的数据。
        
        "prompt": "do something",
        # 指令提示，为非结构化文本。
    }


def _parse_image(image) -> np.ndarray:
    """将输入图像转化为numpy数组并处理其数据类型和形状。"""
    image = np.asarray(image)  # 确保input是numpy数组
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)  # 将浮点数图像缩放至[0,255]并转换为uint8类型
    if image.shape[0] == 3:  
        image = einops.rearrange(image, "c h w -> h w c")  # 如果通道数在第一维，将其改成(H,W,C)格式
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # 模型的动作维度，将用于填充状态和动作。
    action_dim: int

    # 决定使用哪个模型。
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = np.concatenate([data["observation/joint_position"], data["observation/gripper_position"]])
        # 将关节位置和抓手位置拼接成一个状态向量
        
        state = transforms.pad_to_dim(state, self.action_dim)
        # 对状态进行填充，使其维度符合action_dim
        
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        # 解析基础外观图像
        wrist_image = _parse_image(data["observation/wrist_image_left"])
        # 解析腕部图像

        match self.model_type:
            case _model.ModelType.PI0:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # 对于FAST模型，不会屏蔽掉填充图像
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"不支持的模型类型: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            # 创建图像字典，将名称和对应图像关联起来
            
            "image_mask": dict(zip(names, image_masks, strict=True)),
            # 创建图像掩码字典，用于处理模型输入
            
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
            # 将动作添加到inputs中，如果存在

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
            # 将指令加入inputs中，如果存在

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 仅返回前8个维度的动作输出
        return {"actions": np.asarray(data["actions"][:, :8])}
