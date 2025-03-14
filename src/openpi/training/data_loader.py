from collections.abc import Iterator, Sequence
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)  # 定义一个类型变量 T_co，用于表示返回值类型

class Dataset(Protocol[T_co]):
    """抽象接口：用于支持随机访问的数据集。"""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        """根据索引获取数据集中某一元素."""
        raise NotImplementedError("Dataset 类的子类应该实现 __getitem__.")

    def __len__(self) -> int:
        """获取数据集的长度."""
        raise NotImplementedError("Dataset 类的子类应该实现 __len__.")

class DataLoader(Protocol[T_co]):
    """抽象接口：用于数据加载器。"""

    def data_config(self) -> _config.DataConfig:
        """获取当前数据加载器的数据配置."""
        raise NotImplementedError("DataLoader 的子类应该实现 data_config.")

    def __iter__(self) -> Iterator[T_co]:
        """使得 DataLoader 可以被迭代."""
        raise NotImplementedError("DataLoader 的子类应该实现 __iter__.")

class TransformedDataset(Dataset[T_co]):
    """包装一个数据集并应用转换的类."""

    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        """初始化数据集和转换函数。
        
        Parameters:
        - dataset: 原始数据集
        - transforms: 应用在数据集上的转换函数序列
        """
        self._dataset = dataset  # 存储原始数据集
        self._transform = _transforms.compose(transforms)  # 生成多个转换函数组合

    def __getitem__(self, index: SupportsIndex) -> T_co:
        """使用转换函数从数据集中获取变换后的元素."""
        return self._transform(self._dataset[index])  # 返回经过转换的数据

    def __len__(self) -> int:
        """返回原始数据集的长度."""
        return len(self._dataset)

class FakeDataset(Dataset):
    """模拟数据集，用于测试或开发阶段."""

    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        """初始化模拟数据集.
        
        Parameters:
        - model_config: 模型配置信息
        - num_samples: 样本数量
        """
        self._num_samples = num_samples  # 记录样本总数
        self._observation_spec, self._action_spec = model_config.inputs_spec()  # 获取观察和动作规范

    def __getitem__(self, index: SupportsIndex) -> dict:
        """根据索引生成假数据样本."""
        rng = jax.random.key(index.__index__())  # 根据索引生成随机数种子

        # 根据形状规范生成随机数据
        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng  # 使用外部 RNG
            rng, data_rng = jax.random.split(rng)  # 分割随机数生成器以产生新 RNG
            shape = spec.shape[1:]  # 移除批量维度
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)  # 生成浮点数
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)  # 生成整数
            return jnp.zeros(shape=shape, dtype=spec.dtype)  # 返回默认值（全零）

        observation = jax.tree.map(make_from_spec, self._observation_spec)  # 为每个观测生成数据
        action = jax.tree.map(make_from_spec, self._action_spec)  # 为每个动作生成数据

        return {
            **observation.to_dict(),  # 将观测转换为字典
            "actions": action,  # 包含动作信息
        }

    def __len__(self) -> int:
        """返回模拟数据集中的样本数量."""
        return self._num_samples

def create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig) -> Dataset:
    """创建训练所需的数据集.

    Arguments:
    - data_config: 数据配置
    - model_config: 模型配置
    
    Returns:
    - 创建的数据集实例
    """
    repo_id = data_config.repo_id  # 从配置中获取仓库ID
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")  # 如果没有设置工厂ID则抛出异常
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)   # 返回假数据集

    # 加载实际数据集及其元数据
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, local_files_only=data_config.local_files_only)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        },
        local_files_only=data_config.local_files_only,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])  # 添加任务时的Prompt转换

    return dataset  # 返回构建好的数据集


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """对数据集应用各种变换操作.

    Arguments:
    - dataset: 需要变换的数据集
    - data_config: 数据源的配置
    - skip_norm_stats: 是否跳过归一化统计
    
    Returns:
    - 新创建的转化后数据集实例
    """
    norm_stats = {}  # 初始化归一化统计信息
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )  # 如果未找到归一化统计则抛出异常
        norm_stats = data_config.norm_stats  # 获取归一化统计数据

    return TransformedDataset(  # 返回应用有关变换的新数据集
        dataset,
        [
            *data_config.repack_transforms.inputs,  # 重打包转换
            *data_config.data_transforms.inputs,  # 数据转换
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),  # 归一化处理
            *data_config.model_transforms.inputs,  # 模型层面的变换
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """创建训练用的数据加载器.

    Parameters：
    - config: 训练配置
    - sharding: 指定要使用的数据加载器分片。如果为 None，则采用单设备分片
    - skip_norm_stats: 是否跳过数据归一化
    - shuffle: 是否随机打乱数据
    - num_batches: 确定返回的批次数量
    - num_workers: 要使用的工作进程数量，如果为零将主进程执行。

    Returns:
    - 实现了 DataLoader 接口的加载器实例
    """
    data_config = config.data.create(config.assets_dirs, config.model)  # 创建数据配置实例

    dataset = create_dataset(data_config, config.model)  # 创建数据集
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)  # 转换数据集

    # 创建torch数据加载器实例
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),  # 设置局部批尺寸
        sharding=sharding,  # 分片定义
        shuffle=shuffle,  # 随机打乱标志
        num_batches=num_batches,  # 批次设置
        num_workers=num_workers,  # 工作进程数量
        seed=config.seed,  # 随机种子
    )

    class DataLoaderImpl(DataLoader):  # 封装 torch 数据加载器
        def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
            self._data_config = data_config  # 保存数据配置
            self._data_loader = data_loader  # 保存Torch数据加载器

        def data_config(self) -> _config.DataConfig:
            """重写数据配置的方法"""
            return self._data_config  

        def __iter__(self):
            """让 DataLoaderImpl 可迭代"""
            for batch in self._data_loader:
                yield _model.Observation.from_dict(batch), batch["actions"]  # 返回观测对象和对应动作

    return DataLoaderImpl(data_config, data_loader)  # 返回实现了 DataLoader 接口的结果


class TorchDataLoader:
    """使用 PyTorch 来进行数据加载的工具类."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """初始化 PyTorch 数据加载器.

        参数：
        - dataset: 加载的数据集实例
        - local_batch_size: 每个进程的局部批大小
        - sharding: 数据加载器支持的分片
        - shuffle: 是否打乱数据顺序
        - num_batches: 若提供，确定返回的批次数量。超出时会循环遍历数据集
        - num_workers: 工作进程数量，当为零时在主进程中完成加载
        - seed: 用于打乱数据的随机种子
        """
        if jax.process_count() > 1:
            raise NotImplementedError("不支持多进程的数据加载.")
      
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # 默认使用数据并行分片
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding  
        self._num_batches = num_batches 

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")  # 配置并发上下文

        generator = torch.Generator()  
        generator.manual_seed(seed)  
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,  # 设置批大小
            shuffle=shuffle,  # 打乱选项
            num_workers=num_workers,  # 工作进程数
            multiprocessing_context=mp_context, 
            persistent_workers=num_workers > 0,  # 持久性工作进程标识
            collate_fn=_collate_fn,  # 合并函数
            worker_init_fn=_worker_init_fn,  # 工作进程的初始化函数
            drop_last=True,  # 丢弃最后不足一批的数据
            generator=generator,  # 随机生成器
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        """属性获取内部torch数据加载器."""
        return self._data_loader
        
    def __iter__(self):
        """可迭代接口，实现无限轮询"""
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)  # 获取数据迭代器
            while True:
                if self._num_batches is not None and num_items >= self._num_batches: 
                    return  # 达到指定批次退出
                
                try:
                    batch = next(data_iter)  # 获取下一个批次
                except StopIteration:
                    break  # 当前数据集已耗尽，重新开始迭代
                    
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)  # 格式化输出

def _collate_fn(items):
    """将多个batch元素整合成批量numpy数组."""
    # 将所有incoming elements转为numpy arrays后才进行 stack 操作，因为部分可能是JAX arrays
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)

def _worker_init_fn(worker_id: int) -> None:
    """告诉 JAX 在工作进程中不要预分配 GPU 内存。"""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # 不预分配显存
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # 默认为平台选择内存分配器
