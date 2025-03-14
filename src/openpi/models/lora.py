import math
import re

import flax.linen as nn
import flax.struct as struct
import jax.numpy as jnp

import openpi.shared.array_typing as at


@struct.dataclass
class LoRAConfig:
    """LoRA的配置类。"""

    # LoRA的秩（rank）。
    rank: int
    # LoRA的缩放因子。
    alpha: float = 1.0
    # LoRA参数的初始化函数。
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # 启用排名稳定的LoRA：参考文献 https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # 应用于权重的轴，通常是最后两个轴。
    axes: tuple[int, int] = (-2, -1)
    # 在einsum方程中使用的轴标签。必须不在原始方程中出现。
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        # 根据rslora的值计算缩放值。
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


class Einsum(nn.Module):
    """支持LoRA的Einsum模块。可以作为Gemma Einsum的替代品。"""

    # 权重的形状。
    shape: tuple[int, ...]
    # 权重的初始化函数。
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # 如果不为None，则对权重应用LoRA。
    lora_config: LoRAConfig | None = None

    def setup(self):
        # 初始化权重参数w。
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # 设置LoRA参数。
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank  # 更新shape_a以适应LoRA的秩
            shape_b[config.axes[0]] = config.rank  # 更新shape_b以适应LoRA的秩
            self.w_a = self.param("lora_a", config.init_fn, shape_a)  # 初始化LoRA参数a
            self.w_b = self.param("lora_b", config.init_fn, shape_b)  # 初始化LoRA参数b

    @nn.compact
    def __call__(self, eqn: str, x):
        dtype = x.dtype  # 原始数据类型，可能是半精度
        result = jnp.einsum(eqn, x, self.w.astype(dtype))  # 执行einsum操作

        if config := self.lora_config:
            # 如果存在LoRA配置，则生成相应的einsum方程并计算LoRA部分。
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))  # 计算LoRA的第一部分
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))  # 计算LoRA的第二部分
            result = result + lora * config.scaling_value  # 将LoRA结果加到最终结果上

        return result

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        # 创建LoRA所需的einsum方程。
        if "L" in eqn:
            raise ValueError(f"L已经在方程中: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"不支持的einsum方程: {eqn}")
        lhs, rhs, out = m.groups()  # 分解方程的左侧、右侧和输出部分

        assert self.lora_config is not None
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)  # 获取LoRA的标签
        label = self.lora_config.label

        # 替换标签以创建新的einsum方程
        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b


class FeedForward(nn.Module):
    """前馈神经网络模块。"""

    features: int
    hidden_dim: int
    # 如果不为None，则对权重应用LoRA。
    lora_config: LoRAConfig | None = None

    def setup(self):
        # 初始化门控权重和线性权重。
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            # 设置LoRA参数。
            self.w_gating_lora = (
                self.param("gating_einsum_lora_a", self.lora_config.init_fn, (2, self.features, self.lora_config.rank)),
                self.param(
                    "gating_einsum_lora_b", self.lora_config.init_fn, (2, self.lora_config.rank, self.hidden_dim)
                ),
            )
            self.w_linear_lora = (
                self.param("linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features)),
            )

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # 原始数据类型，可能是半精度
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
        )  # 计算门控值
        gate_value = nn.gelu(ff_gate)  # 使用GELU激活函数

        ff1 = self._dot(
            x,
            self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
        )  # 计算前馈层的输入
        activations = gate_value * ff1  # 计算激活值

        outputs = self._dot(activations, self.w_linear, self.w_linear_lora)  # 计算最终输出
        assert outputs.dtype == dtype  # 确保输出的数据类型与输入一致
        return outputs

    def _dot(self, x: at.Array, w: at.Array, lora_weights: tuple[at.Array, at.Array] | None) -> at.Array:
        # 计算矩阵乘法，并考虑LoRA权重。
        base = jnp.dot(x, w.astype(x.dtype))  # 基础矩阵乘法
        if lora_weights is None:
            return base  # 如果没有LoRA权重，返回基础结果
        return base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))  # 加入LoRA的影响