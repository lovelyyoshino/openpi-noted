import dataclasses
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):
    """学习率调度器配置协议，保证具有创建学习率调度的方法。"""
    
    def create(self) -> optax.Schedule: ...  # 创建学习率调度的接口


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """余弦衰减学习率调度，用于带热身（warmup）的学习率调整。"""

    warmup_steps: int = 1_000          # 热身步数；在该期间内线性增加学习率
    peak_lr: float = 2.5e-5            # 学习率峰值
    decay_steps: int = 30_000           # 学习率从峰值衰减到结束值所需的总步数
    decay_lr: float = 2.5e-6            # 衰减后的最终学习率

    def create(self) -> optax.Schedule:
        """创建余弦衰减学习率调度。"""
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),  # 初始学习率
            peak_value=self.peak_lr,                               # 最大学习率
            warmup_steps=self.warmup_steps,                       # 热身步数
            decay_steps=self.decay_steps,                         # 衰减步骤数
            end_value=self.decay_lr,                              # 衰减后最终学习率
        )


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """反平方根衰减学习率调度，也适用于带热身的学习率调整。"""

    warmup_steps: int = 1_000             # 热身步数
    peak_lr: float = 5e-5                  # 学习率峰值
    timescale: float = 10_000              # 时间尺度参数用于计算衰减

    def create(self) -> optax.Schedule:
        """创建反平方根衰减学习率调度。"""
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),  # 初始学习率
                    end_value=self.peak_lr,                                 # 最大学习率
                    transition_steps=self.warmup_steps,                   # 热身的过渡步数
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
                # 根据当前step计算出学习率值的函数
            ],
            [self.warmup_steps],   # 合并计划的时间点
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    """优化器配置协议，确保其具有创建优化器的方法。"""

    def create(
        self,
        lr: optax.ScalarOrSchedule,         # 学习率或学习率调度
        weight_decay_mask: at.PyTree | None = None,  # 权重衰减掩码，可选
    ) -> optax.GradientTransformation: ...  # 返回梯度变换策略


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW优化器配置类。"""

    b1: float = 0.9                      # 一阶矩的指数衰减率
    b2: float = 0.95                     # 二阶矩的指数衰减率
    eps: float = 1e-8                    # 小常量，防止除零错误
    weight_decay: float = 1e-10          # 权重衰减因子
    clip_gradient_norm: float = 1.0       # 梯度裁剪的范数阈值

    def create(
        self,
        lr: optax.ScalarOrSchedule,         # 学习率或调度
        weight_decay_mask: at.PyTree | None = None,  # 权重衰减掩码
    ) -> optax.GradientTransformation:
        """创建AdamW优化器。"""
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )
        
        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)  # 将梯度裁剪和AdamW结合起来


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD优化器配置类。"""

    lr: float = 5e-5    # 学习率
    momentum: float = 0.9 # 动量参数
    nesterov: bool = False  # 是否使用Nesterov加速

    def create(
        self,
        lr: optax.ScalarOrSchedule,          # 学习率或调度
        weight_decay_mask: at.PyTree | None = None,  # 权重衰减掩码
    ) -> optax.GradientTransformation:
        """创建SGD优化器。"""
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"  # 重量衰减不被支持
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)  # 返回SGD实现


def create_optimizer(
    optimizer: OptimizerConfig,               # 优化器配置
    lr_schedule: LRScheduleConfig,            # 学习率调度配置
    weight_decay_mask: at.PyTree | None = None  # 权重衰减掩码，可选
) -> optax.GradientTransformation:
    """根据给定的优化器和学习率调度创建优化器实例。"""
    lr = lr_schedule.create()  # 调用学习率调度的create方法得到具体的学习率
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)  # 创建并返回指定的优化器
