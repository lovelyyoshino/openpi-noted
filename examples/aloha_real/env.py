from typing import List, Optional  # noqa: UP035

import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.aloha_real import real_env as _real_env


class AlohaRealEnvironment(_environment.Environment):
    """一个用于真实硬件上Aloha机器人的环境。"""

    def __init__(
        self,
        reset_position: Optional[List[float]] = None,  # 可选的重置位置，默认为None,Optional表示可以为None
        render_height: int = 224,  # 渲染图像的高度
        render_width: int = 224,   # 渲染图像的宽度
    ) -> None:
        # 初始化真实环境，并设置初始节点和重置位置
        self._env = _real_env.make_real_env(init_node=True, reset_position=reset_position)
        self._render_height = render_height  # 设置渲染高度
        self._render_width = render_width    # 设置渲染宽度

        self._ts = None  # 时间步初始化为None

    @override
    def reset(self) -> None:
        """重置环境并返回新的时间步。"""
        self._ts = self._env.reset()  # 调用环境的重置方法

    @override
    def is_episode_complete(self) -> bool:
        """检查当前回合是否完成。"""
        return False  # 此处简单返回False，表示回合未完成

    @override
    def get_observation(self) -> dict:
        """获取当前观察值，包括状态和图像信息。"""
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")  # 如果时间步未设置，则抛出异常

        obs = self._ts.observation  # 获取当前观察数据
        for k in list(obs["images"].keys()):
            if "_depth" in k:  # 删除深度图像
                del obs["images"][k]

        for cam_name in obs["images"]:
            # 将图像调整为指定大小并转换为uint8格式
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
            )
            # 使用einops重新排列图像维度，从(h, w, c)变为(c, h, w)
            obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")

        return {
            "state": obs["qpos"],  # 返回关节位置作为状态
            "images": obs["images"],  # 返回处理后的图像
        }

    @override
    def apply_action(self, action: dict) -> None:
        """应用给定的动作到环境中。这个在packages/openpi-client/src/openpi_client/runtime/runtime.py中调用，然后其作为子类被调用实现"""
        self._ts = self._env.step(action["actions"])  # 执行动作并更新时间步
