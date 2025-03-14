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

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

# 定义词汇量大小常量
PALIGEMMA_VOCAB_SIZE = 257_152

# 使用数据类定义配置参数
@dataclasses.dataclass
class Config:
    width: int  # 模型宽度
    depth: int  # 模型深度
    mlp_dim: int  # MLP层维度
    num_heads: int  # 注意力头数
    num_kv_heads: int  # 键值对头数
    head_dim: int  # 每个头的维度
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)  # LoRA 配置字典

# 定义可用的模型变体类型
Variant = Literal["dummy", "gemma_300m", "gemma_2b", "gemma_2b_lora"]

def get_config(variant: Variant) -> Config:
    """根据指定的 gemma 变体返回相应的配置。"""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M 参数
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        # 311M 参数
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")  # 抛出未识别变体的异常


@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # 获取原始数据类型，可能为半精度
        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))  # 初始化缩放参数
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # 计算方差，使用float32
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # 标准化输入
        normed_inputs = normed_inputs * (1 + scale)  # 根据学习到的参数进行缩放
        return normed_inputs.astype(dtype)  # 返回与原始数据类型一致的数据


@at.typecheck
class Embedder(nn.Module):
    """嵌入模块。"""

    vocab_size: int  # 词汇表大小
    embed_dim: int  # 嵌入维度

    def setup(self):
        # 设置输入嵌入表的参数
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        # 编码输入，通过查找嵌入表获得嵌入向量
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)  # 对嵌入结果进行缩放
        return x

    def decode(self, x):
        # 解码嵌入向量，通过矩阵乘法重建输入
        return jnp.dot(x, self.input_embedding_table.T)


@at.typecheck
class Attention(nn.Module):
    """注意力模块。"""

    configs: Sequence[Config]  # 注意力头配置序列

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # 所有专家必须共享相同的头维度、头数及键值对头数，以使自注意力工作
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # 确定输入中的数据类型
        
        qkvs = []  # 存储查询（Q）、键（K）和值（V）

        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            
            # 判断是否kv头数等于head数
            if config.num_kv_heads == config.num_heads:
                # 使用爱因斯坦求和记号创建QKV张量
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))  # 计算QKV张量
            else:
                # 单独对Q和KV做处理
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)  # 计算查询Q
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)  # 计算键K和值V
                qkvs.append((q, k, v))

        # 将所有的Q、K、V按批次连接起来
        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        # 应用位置编码
        q = _apply_rope(q, positions=positions)  
        q *= self.configs[0].head_dim ** -0.5  # 缩放Q

        k = _apply_rope(k, positions=positions)  # 同样处理K

        # 检查数据类型
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache  # 从缓存中获取K和V
            k = jnp.concatenate([cache_k, k], axis=1)  # 合并新旧K
            v = jnp.concatenate([cache_v, v], axis=1)  # 合并新旧V

        # 重塑Q以适配后续运算
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)  # 计算logits

        # 验证注意力掩码形状
        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        big_neg = -2.3819763e38  # a large negative value for masking
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)  # 应用mask以限制无效区域

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)  # 计算softmax得到概率分布

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)  # 利用概率加权求和得编码结果
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")  # 重塑编码输出

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))  # 生成最终输出
                start = end
            else:
                out.append(None)

        return out, (k, v)  # 返回输出结果及更新的k/v缓存

@at.typecheck
class FeedForward(nn.Module):
    """前馈模块。"""

    features: int  # 输入特征维度
    hidden_dim: int  # 隐藏层维度

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # 原始数据类型，可以是半精度
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)  # 初始化门控权重
        
        ff_gate = jnp.dot(x, w_gating[0])  # 计算门控值
        gate_value = nn.gelu(ff_gate)  # 应用GELU激活函数

        ff1 = jnp.dot(x, w_gating[1])  # 计算前馈网络的输出
        activations = gate_value * ff1  # 使用门控值加权前馈输出

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)  # 初始化线性变换权重
        
        outputs = jnp.dot(activations, w_linear)  # 计算最终输出
        assert outputs.dtype == dtype  # 确保输出的数据类型与输入一致
        return outputs


@at.typecheck
class Block(nn.Module):
    """Transformer块。"""

    configs: Sequence[Config]  # 配置序列

    dropout: float = 0.0  # Dropout比率
    dropout_bdims: tuple[int, ...] = ()  # 每个浮点数独立丢弃的维度

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, decode, deterministic=True):  # noqa: FBT002
        xs = sharding.activation_sharding_constraint(xs)  # 限制激活的分片
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x  # 定义Dropout操作

        attn = Attention(configs=self.configs, name="attn")  # 创建注意力机制实例

        pre_attn = []  # 存储预处理后的输入
        for i, x in enumerate(xs):
            if x is not None:
                x = RMSNorm(name=_name("pre_attention_norm", i))(x)  # 对输入进行归一化
            pre_attn.append(x)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)  # 限制预处理激活的分片
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)  # 执行注意力机制
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)  # 应用Dropout
        post_attn = sharding.activation_sharding_constraint(post_attn)  # 限制后处理激活的分片
        xs = jax.tree.map(lambda x, y: x + y, xs, post_attn)  # 残差连接
        xs = sharding.activation_sharding_constraint(xs)  # 再次限制激活的分片

        out = []  # 存储输出结果
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x = RMSNorm(name=_name("pre_ffw_norm", i))(x)  # 对输入进行归一化
                x = lora.FeedForward(  # 调用前馈网络
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            out.append(x)

        out = sharding.activation_sharding_constraint(out)  # 限制输出激活的分片

        out = jax.tree.map(lambda x: drop(x, deterministic), out)  # 应用Dropout
        xs = jax.tree.map(lambda x, y: x + y, xs, out)  # 残差连接
        xs = sharding.activation_sharding_constraint(xs)  # 最终限制激活的分片

        return xs, kv_cache  # 返回更新后的输入和kv缓存


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


@at.typecheck
class Module(nn.Module):
    """Transformer模型，支持不同token的不同权重混合。"""

    configs: Sequence[Config]  # 每个专家的配置列表
    embed_dtype: str  # 嵌入数据类型

    dropout: float = 0.0  # Dropout比率
    dropout_bdims: tuple[int, ...] = ()  # 每个浮点数独立丢弃的维度

    def setup(self):
        # 所有专家必须具有相同的深度
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # 仅为第一个专家创建嵌入器
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5,),  # 0=self, 5=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast),  # 0=kv_cache, 1=positions, 2=mask, 3=decode
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)  # 将tokens编码为嵌入向量并转换为指定数据类型

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],  # 每个专家的token数组或None（如果该专家不运行）
        positions: at.Int[at.Array, "b t"],  # token的位置
        mask: at.Bool[at.Array, "b t s"],  # 注意力掩码
        *,
        kv_cache: KVCache | None = None,  # 键值缓存
        deterministic: bool = True,  # 是否确定性推理
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)  # 转换嵌入数据类型
        mask = jnp.asarray(mask)[:, None, :, :]  # 扩展掩码维度

        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, deterministic)  # 通过层传递嵌入

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)  # 确保所有嵌入的dtype一致

        return [f(e) if e is not None else e for f, e in zip(self.final_norms, embedded, strict=True)], kv_cache  # 返回经过最终归一化的嵌入和kv缓存

    def init(self):
        """初始化所有参数的便利方法，由于linen的特殊性而必要。"""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))  # 初始化嵌入
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],  # 为每个专家初始化零张量
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),  # 初始化位置张量
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),  # 初始化掩码张量
        )


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """将RoPE位置应用于x [B, L]到x [B, L, H, D]。"""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)  # 频率指数
    timescale = max_wavelength**freq_exponents  # 时间尺度
    radians = positions[..., None] / timescale[None, None, :]  # 将位置映射到弧度
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)  # 计算正弦和余弦
    x1, x2 = jnp.split(x, 2, axis=-1)  # 将x拆分为两部分
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)  # 应用RoPE公式
    assert res.dtype == jnp.float32
    # 根据原始bigvision实现，允许RoPE上升到float32，然后在推理模式下立即降级回缓存dtype
    return res.astype(x.dtype)  # 返回与输入相同数据类型的结果


def _name(name, i):
    # 我们以这种方式命名层，因为我们希望第一个专家的权重没有后缀（例如，“attn”），这样它们可以无缝加载现有的PaliGemma检查点。
    # 后续专家将有后缀（例如，“attn_1”），其权重将从头开始初始化。在实践中，我们只使用两个专家 -- PaliGemma和动作专家。
    if i == 0:
        return name
    return f"{name}_{i}"
