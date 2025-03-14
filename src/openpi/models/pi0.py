import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """根据 big_vision 进行改编。

    Tokens 可以关注有效输入 tokens，其 cumulative mask_ar 小于或等于它们的值。
    这样 `mask_ar` bool[?B, N] 可用于设置几种类型的注意力，例如：

      [[1 1 1 1 1 1]]: 纯因果注意力。

      [[0 0 0 1 1 1]]: 前缀-lm 注意力。前 3 个 token 可以相互关注，最后 3 个 token 有因果注意力。第一个条目也可以是 1 而不改变行为。

      [[1 0 1 0 1 0 0 1 0 0]]: 在 4 个块之间的因果注意力。一个块的 tokens 可以关注所有之前的块和同一块上的所有 tokens。

    参数：
      input_mask: bool[B, N] 如果是输入的一部分则为真，如果是填充则为假。
      mask_ar: bool[?B, N] 在其上方的 token 不能依赖于该 token 的掩码，其中为真表示无法依赖，假表示与前一个 token 共享相同的注意力掩码。
    """
    # 将 mask_ar 广播到与 input_mask 相同的形状
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    # 计算 mask_ar 的累积和，输出的形状默认是三维的，对应的是 batch, seq_len, seq_len
    cumsum = jnp.cumsum(mask_ar, axis=1)
    # 创建注意力掩码，允许当前 token 关注之前的 token，cumsum [:, None, :] <= cumsum [:, :, None] 生成一个三角形矩阵, 用于掩码
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    # 生成有效掩码，仅在 input_mask 为真时有效
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck # 检查输入的类型
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """计算标量位置的正弦-余弦位置嵌入向量。"""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) 必须能被 2 整除")

    # 计算周期范围内的分数
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    # 返回正弦和余弦的拼接结果
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # 设置模型特定的默认值。
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        # 定义图像规格
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },# 用于指定图像的掩码
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),# 用于指定状态
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),# 用于指定标记化提示
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),# 用于指定标记化提示的掩码
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """返回基于模型配置的冻结过滤器。"""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")# 匹配所有包含llm的参数
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")# 匹配所有包含llm_1的参数
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # 如果仅冻结 gemma 参数，则排除动作专家参数。
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # 如果使用了任何 lora，则排除所有 lora 参数。
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):
    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        # 初始化 Pi0 模型。
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        
        # 获取 PaliGemma 和动作专家的配置
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        
        # TODO: 用 NNX 重写 gemma。目前使用桥接。
        # 将 Gemma 模块初始化为 LLM（大语言模型）
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")  # 懒加载初始化

        # 初始化图像处理模块
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        # 将 LLM 和图像模块存储在字典中
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # 定义多个线性投影层以处理状态和动作相关的信息
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        # 嵌入图像
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # 图像 tokens 彼此之间可以关注
            ar_mask += [False] * image_tokens.shape[1]

        # 添加语言（即标记化输入）
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # 图像和语言输入之间完全关注
            ar_mask += [False] * tokenized_inputs.shape[1]
            
        # 合并所有 token，mask 和 ar_mask
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)  # 转换为数组
        
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        
        # 添加单个状态 token 
        state_token = self.state_proj(obs.state)[:, None, :]  # 状态通过状态投影得到
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))  # 全部掩码为真
        # 图像/语言输入不关注状态或动作
        ar_mask += [True]

        # 使用正弦-余弦位置编码嵌入时间步
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        
        # 使用 MLP 混合时间步 + 动作信息
        action_tokens = self.action_in_proj(noisy_actions)  # 动作信息转化为 tokens
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)  # 扩展至 horizon
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)  # Swish 激活函数
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)  # 输出
            
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # 图像/语言/状态输入不关注动作 tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        # 合并所有 token，mask 和 ar_mask
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)  # 分割随机种子
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)  # 预处理观察数据

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)  # 正态分布噪声
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001  # Beta 分布生成时间值
        time_expanded = time[..., None, None]  # 扩展维度到匹配

        x_t = time_expanded * noise + (1 - time_expanded) * actions  # 计算干扰样本
        u_t = noise - actions  # 真正的目标信号

        # 一次性执行前缀 + 后缀的大规模前向传递
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)  # 合并 mask
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)  # 合并 ar_mask

        attn_mask = make_attn_mask(input_mask, ar_mask)  # 构建注意力掩码
        positions = jnp.cumsum(input_mask, axis=1) - 1  # 计算 token 的位置信息
        
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )
        
        # 从后缀输出中获取最后时刻的动作的预测
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # 返回平方损失均值
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)

        # 注意：根据扩散文献中的通用约定进行采样，其中 t=1 是噪声，t=0 是目标分布
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))  # 初始化噪声

        # 首先用前缀的前向传递填充 KV 缓存
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)  # 为前缀构建注意力掩码
        positions = jnp.cumsum(prefix_mask, axis=1) - 1  # 前缀同样需要位置信息
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )

            # 构建后缀 attention 掩码
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # 为后续的解码提供完整的 attention 掩码
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            
            # 更新后缀 token 的位置索引
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None
            # 从后缀输出计算下一个动作
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt  # 更新状态

        def cond(carry):
            x_t, time = carry
            # 避免浮点误差导致的问题，继续直到时间范围超出
            return time >= -dt / 2

        # 通过循环更新状态 x_0 和时间
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0  # 返回最终生成的动作
