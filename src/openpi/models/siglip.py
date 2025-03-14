# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A refactored and simplified ViT adoptation for Pi, taken from big_vision."""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp #这个是jax的numpy模块，用法和numpy基本一致，但是jax的numpy支持GPU加速
import numpy as np

import openpi.training.sharding as sharding


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """根据 MoCo v3 的逻辑生成二维位置嵌入（sine-cosine）。"""
    y, x = jnp.mgrid[:h, :w]  # 创建一个网格，y 和 x 分别表示高度和宽度的坐标

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"  # 确保宽度是4的倍数
    omega = jnp.arange(width // 4) / (width // 4 - 1)  # 计算频率因子
    omega = 1.0 / (temperature**omega)  # 根据温度调整频率
    y = jnp.einsum("m,d->md", y.flatten(), omega)  # 将 y 坐标与频率相乘
    x = jnp.einsum("m,d->md", x.flatten(), omega)  # 将 x 坐标与频率相乘
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)  # 拼接正弦和余弦值
    return jnp.asarray(pe, dtype)[None, :, :]  # 返回形状为(1, h*w, width)的数组


# 这里的self是一个Module实例，它是一个继承自nn.Module的类
def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    """获取不同类型的位置嵌入。"""
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),  # 使用正态分布初始化参数
            (1, np.prod(seqshape), width),  # 参数形状
            dtype,
        ) # 这里的self.param是nn.Module中的参数定义函数，用于定义模型参数。参数包括参数名、初始化方法、参数形状和数据类型
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)  # 调用上面的函数生成sinusoidal位置嵌入
    raise ValueError(f"Unknown posemb type: {typ}")  # 抛出未知类型错误


class MlpBlock(nn.Module):
    """Transformer中的MLP/前馈块。"""

    mlp_dim: int | None = None  # 默认设置为输入维度的4倍
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact # 用于定义模块的前向传播
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """应用Transformer MlpBlock模块。"""
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),  # 权重初始化
            "bias_init": nn.initializers.normal(stddev=1e-6),  # 偏置初始化
        }

        _, _, d = x.shape  # 获取输入x的形状 n,l,d
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)  # 全连接层
        x = nn.gelu(x)  # 激活函数
        x = nn.Dropout(rate=self.dropout)(x, deterministic)  # 应用dropout
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)  # 输出全连接层


class Encoder1DBlock(nn.Module):
    """单个Transformer编码器块（MHSA + MLP）。"""

    mlp_dim: int | None = None  # 默认设置为输入维度的4倍
    num_heads: int = 12  # 多头注意力的头数
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)  # 限制激活的分片
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)  # 层归一化
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)  # 自注意力机制
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)  # 应用dropout
        x = out["+sa"] = x + y  # 残差连接

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)  # 再次进行层归一化
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)  # 应用MLP块
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)  # 应用dropout
        x = out["+mlp"] = x + y  # 残差连接
        x = sharding.activation_sharding_constraint(x)
        return x, out  # 返回输出和中间结果


class Encoder(nn.Module):
    """用于序列到序列翻译的Transformer模型编码器。"""

    depth: int  # 编码器深度
    mlp_dim: int | None = None  # 默认设置为输入维度的4倍
    num_heads: int = 12  # 多头注意力的头数
    dropout: float = 0.0
    scan: bool = False  # 是否使用scan
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}

        if self.scan:
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),  # 0=self, 2=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic)
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)  # 保存每个块的输出
        else:
            # 输入编码器
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)  # 每个块的前向传播
            out["pre_ln"] = x  # 最后一个块的别名，但没有编号。

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out  # 返回最终输出和所有块的输出


class MAPHead(nn.Module):
    """多头注意力池化。"""

    mlp_dim: int | None = None  # 默认设置为输入维度的4倍
    num_heads: int = 12  # 多头注意力的头数
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape  # n,l,d
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)  # 初始化探针
        probe = jnp.tile(probe, [n, 1, 1])  # 扩展探针以匹配批量大小

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)  # 应用多头注意力

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)  # 层归一化
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype_mm)(y)  # 残差连接
        return x[:, 0]  # 返回第一个token的输出


class _Module(nn.Module):
    """ViT模型。"""

    num_classes: int | None = None  # 类别数量
    patch_size: Sequence[int] = (16, 16)  # 图像切片大小
    width: int = 768  # 特征宽度
    depth: int = 12  # 编码器深度
    mlp_dim: int | None = None  # 默认设置为输入维度的4倍
    num_heads: int = 12  # 多头注意力的头数
    posemb: str = "learn"  # 可以是“learn”或“sincos2d”
    rep_size: int | bool = False  # 表示是否需要代表性大小
    dropout: float = 0.0  # dropout比率
    pool_type: str = "gap"  # 池化类型，可以是“map”或“tok”
    head_zeroinit: bool = True  # 是否将头部权重初始化为零
    scan: bool = False  # 是否使用scan
    remat_policy: str = "nothing_saveable"  # 重计算策略
    dtype_mm: str = "float32"  # 数据类型

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}

        # Kevin edit: 在float32中提取补丁和位置嵌入，
        # 因为我觉得这样更安全。
        image = jnp.asarray(image, jnp.float32)

        # 补丁提取
        x = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image)

        n, h, w, c = x.shape  # 获取补丁的形状
        x = jnp.reshape(x, [n, h * w, c])  # 重塑为(n, h*w, c)

        # 在添加额外token之前添加位置嵌入。
        x = out["with_posemb"] = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)  # 初始化CLS token
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)  # 添加CLS token

        n, _, c = x.shape  # 更新形状 n,l,d
        x = nn.Dropout(rate=self.dropout)(x, not train)  # 应用dropout

        # Kevin edit: 现在转换回dtype_mm（可能是半精度）
        x = x.astype(self.dtype_mm)

        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, deterministic=not train)  # 编码过程
        encoded = out["encoded"] = x  # 保存编码后的输出

        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype_mm,
            )(x)  # 应用MAP头
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)  # 全局平均池化
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]  # 选择第一个token
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]  # 选择第一个token
            encoded = encoded[:, 1:]  # 去掉CLS token
        elif self.pool_type == "none":
            pass  # 不做任何操作
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")  # 抛出未知池化类型错误

        x_2d = jnp.reshape(encoded, [n, h, w, -1])  # 重塑编码后的输出

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size  # 设置代表性大小
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")  # 定义全连接层
            # 注意：过去我们在pre_logits中不包括tanh。
            # 对于少样本，这应该没什么大不了，因为它会被白化。
            x_2d = nn.tanh(hid(x_2d))  # 应用tanh激活
            x = nn.tanh(hid(x))  # 应用tanh激活

        out["pre_logits_2d"] = x_2d  # 保存2D预处理输出
        out["pre_logits"] = x  # 保存预处理输出

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}  # 如果需要，将头部权重初始化为零
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)  # 定义分类头
            x_2d = out["logits_2d"] = head(x_2d)  # 计算2D logits
            x = out["logits"] = head(x)  # 计算logits

        return x, out  # 返回最终输出和所有中间结果


def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """工厂函数，因为linen真的不喜欢我正在做的事情！"""
    return _Module(num_classes, **{**decode_variant(variant), **kw})  # 创建并返回_Module实例


def decode_variant(variant):
    """将字符串如"B"或"B/32"转换为参数字典。"""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")  # 分割变体和补丁信息
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # 参考文献：https://arxiv.org/abs/2106.04560的表2。
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        # pylint:enable=line-too-long
        **patch,
    }
