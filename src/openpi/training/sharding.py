import contextlib
import logging

import jax
import numpy as np

BATCH_AXIS = "batch"  # 批处理维度
FSDP_AXIS = "fsdp"    # FSDP 维度（Fully Sharded Data Parallel）
# 在 FSDP 中，我们将在批处理和 FSDP 维度之间分割数据。
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)  # 数据的轴，表示我们将使用这两个维度进行分片


class _MeshState:
    active_mesh: jax.sharding.Mesh | None = None  # 存储当前活动的网格状态，如果没有则为 None


def make_mesh(num_fsdp_devices: int) -> jax.sharding.Mesh:
    """创建一个 JAX 网格，以实现分布式训练。

    参数：
        num_fsdp_devices: 每个计算节点使用的 FSDP 设备数量。

    返回：
        一个用于拆分数据的 JAX 网格对象。

    抛出：
        ValueError 如果总设备数不能被指定的 FSDP 设备数量整除。
    """
    if jax.device_count() % num_fsdp_devices != 0:
        raise ValueError(
            f"Number of devices {jax.device_count()} must be divisible by the number of FSDP devices {num_fsdp_devices}."
        )
    mesh_shape = (jax.device_count() // num_fsdp_devices, num_fsdp_devices)  # 创建网格形状
    return jax.make_mesh(mesh_shape, (BATCH_AXIS, FSDP_AXIS))  # 根据形状生成网格


@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh):
    """设置全局网格的上下文管理器。用来维护对全局网格引用，
    此方法仅在 `activation_sharding_constraint` 中使用。

    抛出：
        ValueError 如果试图嵌套使用此管理器。
    """
    if _MeshState.active_mesh is not None:  # 检查是否已经存在活动网格
        raise ValueError("Cannot nest set_mesh context managers.")
    _MeshState.active_mesh = mesh  # 更新当前活动的网格
    try:
        yield  # 保持上下文块中执行的代码可以访问这个网格
    finally:
        _MeshState.active_mesh = None  # 清理活动网格


def activation_sharding_constraint(pytree):
    """根据当前激活的网格，为给定的 pytree 应用分片约束。

    参数：
        pytree: 要应用分片的 Python 树结构，可以是任意类型，但通常包含数组。

    返回：
        应用分片约束后的 pytree。
    """
    if _MeshState.active_mesh is None:  # 如果没有激活的网格，则返回原始 pytree
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec(DATA_AXIS))
    )  # 使用当前网格和分片规范调整 pytree 的分片约束


def fsdp_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    *,
    min_size_mbytes: int = 4,  # 最小分片数组大小（以 MiB 为单位）
    log: bool = False,          # 是否记录分片决策
):
    """基于网格形状对 pytree 中数组应用 FSDP 分片。

    参数：
        pytree: 待应用分片指定的 pytree，仅考虑具有 .shape 属性的数组类型。
        mesh: 用于对 pytree 应用分片的网格。
        min_size_mbytes: 要考虑分片的最小数组大小，如果数组小于此值，将会复制该数组。
        log: 如果为真，将记录所有正在考虑分片的数组的分片决定。

    返回：
        已分片的 pytree。
    """
    min_size_bytes = min_size_mbytes * 2**20  # 将 min_size_mbytes 转换为字节单位

    def _shard_arr(kp, array: jax.ShapeDtypeStruct):
        """实际处理分片逻辑的内部函数。

        参数：
            kp: pytree 中当前元素的路径信息。
            array: 当前要处理的数组及其形状和数据类型信息。

        返回：
            适当的数据分片配置。
        """
        # 如果不打算使用 FSDP，则复制所有内容以避免冗余的日志信息
        if mesh.shape[FSDP_AXIS] == 1:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())  # 不分片直接依赖单一 FSDP 设备
        # 复制标量和向量数组
        if not hasattr(array, "shape"):
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if len(array.shape) < 2:  # 小于二级（如，标量、向量）数组也只复制
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # 复制较小数组
        if (arr_size := np.prod(array.shape) * np.dtype(array.dtype).itemsize) < min_size_bytes:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # 对矩阵和更大的张量，在能够被 FSDP 维度整除的最大轴上进行分片
        axes = np.argsort(array.shape)[::-1]  # 获取各个维度索引按大小逆序排列
        spec = [None] * len(axes)  # 初始化分片规格
        for i in axes:
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:  # 找到可以分片的特定维度
                if log:  # 日志记录
                    logging.info(
                        f"Sharding {jax.tree_util.keystr(kp)} of shape {array.shape} ({arr_size / 2**20:.2f} MiB) along axis {i}"
                    )
                spec[i] = FSDP_AXIS
                return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))  # 返回已配置的分片
        
        # 如果没有找到有效的分片，则进行复制，并发出警告
        if log:
            logging.warning(
                f"Could not find a valid sharding for {jax.tree_util.keystr(kp)} of shape {array.shape} with mesh of shape {mesh.shape}"
            )
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())  # 默认情况仍然是复制 

    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)  # 遍历 pytree 并应用内部分片逻辑
