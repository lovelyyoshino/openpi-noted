# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ViT implementation adapted from https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py."""

from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models import resnet as models_resnet

Array = Any
PRNGKey = Any
Shape = tuple[int]
Dtype = Any


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""
    
    @nn.compact
    def __call__(self, x):
        return x  # 返回输入，不做任何变换


class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs.

    Attributes:
      posemb_init: positional embedding initializer.
    """

    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]  # 用于初始化位置嵌入的函数
    param_dtype: Dtype = jnp.float32  # 参数的数据类型

    @nn.compact
    def __call__(self, inputs):
        """Applies the AddPositionEmbs module.

        Args:
          inputs: Inputs to the layer.

        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape 是 (batch_size, seq_len, emb_dim)
        assert inputs.ndim == 3, f"Number of dimensions should be 3, but it is: {inputs.ndim}"
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])  # 定义位置嵌入的形状
        pe = self.param("pos_embedding", self.posemb_init, pos_emb_shape, self.param_dtype)  # 初始化位置嵌入
        return inputs + pe  # 将位置嵌入加到输入上


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int  # MLP层的维度
    dtype: Dtype = jnp.float32  # 数据类型
    param_dtype: Dtype = jnp.float32  # 参数数据类型
    out_dim: int | None = None  # 输出维度
    dropout_rate: float = 0.1  # Dropout比率
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()  # 权重初始化方法
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)  # 偏置初始化方法

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim  # 确定输出维度
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(  # pytype: disable=wrong-arg-types
            inputs
        )
        x = nn.gelu(x)  # 应用GELU激活函数
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)  # 应用Dropout
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(  # pytype: disable=wrong-arg-types
            x
        )
        return nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)  # 再次应用Dropout


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """

    mlp_dim: int  # MLP的维度
    num_heads: int  # 注意力头的数量
    dtype: Dtype = jnp.float32  # 数据类型
    dropout_rate: float = 0.1  # Dropout比率
    attention_dropout_rate: float = 0.1  # 注意力机制中的Dropout比率

    @nn.compact
    def __call__(self, inputs, deterministic):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """

        # Attention block.
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"  # 检查输入维度
        x = nn.LayerNorm(dtype=self.dtype)(inputs)  # 对输入进行层归一化
        x = nn.MultiHeadDotProductAttention(
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            num_heads=self.num_heads,
            force_fp32_for_softmax=True,
        )(x, x)  # 计算多头注意力
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)  # 应用Dropout
        x = x + inputs  # 残差连接

        # MLP block.
        y = nn.LayerNorm(dtype=self.dtype)(x)  # 对残差结果进行层归一化
        y = MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic
        )  # 应用MLP块

        return x + y, None  # 返回经过处理的结果和None（可能用于后续操作）


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
      num_layers: number of layers
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate in self attention.
    """

    dtype: jax.typing.DTypeLike  # 数据类型
    num_layers: int  # 层数
    mlp_dim: int  # MLP的维度
    num_heads: int  # 注意力头的数量
    dropout_rate: float = 0.1  # Dropout比率
    attention_dropout_rate: float = 0.1  # 注意力机制中的Dropout比率
    add_position_embedding: bool = True  # 是否添加位置嵌入

    @nn.compact
    def __call__(self, x, *, train):
        """Applies Transformer model on the inputs.

        Args:
          x: Inputs to the layer.
          train: Set to `True` when training.

        Returns:
          output of a transformer encoder.
        """
        assert x.ndim == 3  # (batch, len, emb)

        if self.add_position_embedding:
            x = AddPositionEmbs(
                posemb_init=nn.initializers.normal(stddev=0.02),  # 从BERT中获取的初始化方式
                name="posembed_input",
            )(x)  # 添加位置嵌入
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)  # 应用Dropout

        x = x.astype(self.dtype)  # 转换为指定的数据类型
        # Input Encoder
        block = nn.remat(Encoder1DBlock, prevent_cse=False, static_argnums=(2,))
        x, _ = nn.scan(
            block,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=nn.broadcast,
            length=self.num_layers,
        )(
            name="encoderblock",
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype,
            num_heads=self.num_heads,
        )(x, not train)  # 执行多个编码器块
        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype)(x)  # 最后的层归一化


class VisionTransformer(nn.Module):
    """VisionTransformer."""

    dtype: jax.typing.DTypeLike  # 数据类型
    num_classes: int  # 类别数量
    patches: Any  # 图像分块信息
    transformer: Any  # Transformer相关配置
    hidden_size: int  # 隐藏层大小
    resnet: Any | None = None  # 可选的ResNet模型
    representation_size: int | None = None  # 表示层大小
    classifier: str = "token"  # 分类器类型
    head_bias_init: float = 0.0  # 分类头偏置初始化值
    encoder: type[nn.Module] = Encoder  # 编码器模块
    model_name: str | None = None  # 模型名称

    @nn.compact
    def __call__(self, inputs, *, train):
        x = inputs
        # (Possibly partial) ResNet root.
        if self.resnet is not None:
            width = int(64 * self.resnet.width_factor)

            # Root block.
            x = models_resnet.StdConv(
                features=width, kernel_size=(7, 7), strides=(2, 2), use_bias=False, name="conv_root"
            )(x)  # 根卷积层
            x = nn.GroupNorm(name="gn_root")(x)  # 组归一化
            x = nn.relu(x)  # ReLU激活
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")  # 最大池化

            # ResNet stages.
            if self.resnet.num_layers:
                x = models_resnet.ResNetStage(
                    block_size=self.resnet.num_layers[0], nout=width, first_stride=(1, 1), name="block1"
                )(x)  # 第一个ResNet阶段
                for i, block_size in enumerate(self.resnet.num_layers[1:], 1):
                    x = models_resnet.ResNetStage(
                        block_size=block_size, nout=width * 2**i, first_stride=(2, 2), name=f"block{i + 1}"
                    )(x)  # 后续ResNet阶段

        n, h, w, c = x.shape  # 获取当前张量的形状

        # We can merge s2d+emb into a single conv; it's the same.
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=self.patches.size,
            strides=self.patches.size,
            padding="VALID",
            name="embedding",
        )(x)  # 卷积层，将图像转换为嵌入

        # Here, x is a grid of embeddings.

        # (Possibly partial) Transformer.
        if self.transformer is not None:
            n, h, w, c = x.shape
            x = jnp.reshape(x, [n, h * w, c])  # 重塑为(batch_size, seq_length, embedding_dim)

            # If we want to add a class token, add it here.
            if self.classifier in ["token", "token_unpooled"]:
                cls = self.param("cls", nn.initializers.zeros, (1, 1, c))  # 创建类别标记
                cls = jnp.tile(cls, [n, 1, 1])  # 扩展类别标记
                x = jnp.concatenate([cls, x], axis=1)  # 将类别标记与嵌入拼接

            x = self.encoder(name="Transformer", **self.transformer, dtype=self.dtype)(x, train=train)  # 应用编码器，其中**代表传递字典解包为关键字参数。所以在你给出的代码中，**self.transformer 表示将 self.transformer 字典中的所有键值对作为关键字参数传递给 self.encoder 函数。

        # 根据分类器类型选择输出
        if self.classifier == "token":
            x = x[:, 0]  # 使用类别标记作为输出
        elif self.classifier == "gap":
            x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # 全局平均池化
        elif self.classifier in ["unpooled", "token_unpooled"]:
            pass  # 不做处理
        else:
            raise ValueError(f"Invalid classifier={self.classifier}")  # 抛出错误

        if self.representation_size is not None:
            x = nn.Dense(features=self.representation_size, name="pre_logits")(x)  # 如果有表示层，则通过全连接层
            x = nn.tanh(x)  # Tanh激活
        else:
            x = IdentityLayer(name="pre_logits")(x)  # 否则直接返回

        if self.num_classes:
            x = nn.Dense(
                features=self.num_classes,
                name="head",
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.constant(self.head_bias_init),
            )(x)  # 最终输出层
        return x  # 返回最终输出
