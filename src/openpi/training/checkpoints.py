import concurrent.futures as futures  # 导入并发处理模块
import dataclasses                     # 导入数据类模块
import logging                         # 导入日志模块
from typing import Protocol            # 从typing导入Protocol，用于类型提示

from etils import epath               # 导入文件路径工具模块
import jax                            # 导入jax库用于数值计算
import orbax.checkpoint as ocp        # 导入orbax检查点模块，用于模型存储

from openpi.shared import array_typing as at  # 导入数组类型模块
import openpi.shared.normalize as _normalize  # 导入归一化工具模块
import openpi.training.data_loader as _data_loader  # 导入数据加载器模块
import openpi.training.utils as training_utils      # 导入训练实用工具模块


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    """
    初始化检查点目录，准备保存训练状态等信息。
    
    参数:
    checkpoint_dir : epath.Path 或 str - 检查点目录。
    keep_period : int 或 None - 保留周期，若为None则不限制。
    overwrite : bool - 是否覆盖已存在的目录。
    resume : bool - 是否恢复之前的训练。

    返回：
    tuple[CheckpointManager, bool] - 返回检查点管理器和一个布尔值指示是否恢复训练。
    """
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()  # 将传入路径解析为绝对路径
    resuming = False      # 默认未恢复状态
    if checkpoint_dir.exists():   # 检查目录是否存在
        if overwrite:  # 如果选择重写
            checkpoint_dir.rmtree()  # 删除已有目录
            checkpoint_dir.mkdir(parents=True, exist_ok=True)  # 创建新目录
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")  # 打印清理日志
        elif resume:  # 如果选择恢复
            resuming = True  # 设置恢复标记
        else:  # 若既不覆盖也不恢复，则抛出异常
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)  # 确保创建目录
    
    # 参考https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.checkpoint_manager.html这个链接
    mngr = ocp.CheckpointManager(  # 创建检查点管理器实例
        checkpoint_dir,
        item_handlers={  # 定义不同项的处理方式，这些需要和下面的保存的参数对应
            "assets": CallbackHandler(),  # 特殊处理资产
            "train_state": ocp.PyTreeCheckpointHandler(),  # 使用PyTree管理训练状态
            "params": ocp.PyTreeCheckpointHandler(),  # 使用PyTree管理参数
        },
        options=ocp.CheckpointManagerOptions(  # 配置选项
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),  # 异步选项超时设定
        ),
    )

    # 特别情况处理在恢复训练但没有达到第一个检查点保存的情况下，不应尝试恢复。其中tuple(mngr.all_steps()) in [(), (0,)]表示检查 mngr.all_steps() 的返回值是否为一个空元组 () 或者 仅包含一个元素 0 的元组 (0,)
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]: 
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False  # 重置恢复标记

    return mngr, resuming  # 返回管理器和恢复状态


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    """
    保存当前训练状态和相关资产。

    参数:
    checkpoint_manager : CheckpointManager - 检查点管理器实例。
    state : TrainState - 当前训练状态。
    data_loader : DataLoader - 数据加载器。
    step : int - 当前训练步骤编号。
    """
    def save_assets(directory: epath.Path):
        # 保存归一化统计数据（norm stats）。
        data_config = data_loader.data_config()  # 获取数据配置
        norm_stats = data_config.norm_stats  # 提取归一化统计信息
        if norm_stats is not None and data_config.asset_id is not None:  # 检查是否有归一化统计和资产ID
            _normalize.save(directory / data_config.asset_id, norm_stats)  # 保存归一化统计到指定目录

    # 禁用类型检查以便支持更灵活的数据操作
    with at.disable_typechecking():
        train_state, params = _split_params(state)  # 分离正在使用的参数与训练状态
    items = {
        "assets": save_assets,  # 指定要保存的资源
        "train_state": train_state,  # 保存训练状态
        "params": {"params": params},  # 包装参数
    }
    checkpoint_manager.save(step, items)  # 调用检查点管理器保存状态


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    """
    恢复先前保存的训练状态。

    参数:
    checkpoint_manager : CheckpointManager - 检查点管理器实例。
    state : TrainState - 当前训练状态。
    data_loader : DataLoader - 数据加载器。
    step : int 或 None - 要恢复的步骤，若为None将恢复最近一步。

    返回：
    TrainState - 经恢复的训练状态。
    """
    del data_loader  # 删除未使用的数据加载器引用

    with at.disable_typechecking():  # 禁用类型检查以便灵活操作
        train_state, params = _split_params(state)  # 分离培训状态及其参数
        restored = checkpoint_manager.restore(  # 从检查点恢复状态和参数
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    # 合并恢复后的参数与训练状态
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    """
    加载给定资产ID对应的归一化统计数据。

    参数:
    assets_dir : epath.Path 或 str - 资产目录。
    asset_id : str - 资产ID。

    返回：
    dict或None - 返回加载的归一化统计字典；如果未找到，则返回None。
    """
    norm_stats_dir = epath.Path(assets_dir) / asset_id  # 构建归一化统计目录路径
    norm_stats = _normalize.load(norm_stats_dir)  # 加载归一化统计数据
    logging.info(f"Loaded norm stats from {norm_stats_dir}")  # 日志记录加载的信息
    return norm_stats  # 返回归一化统计字典


class Callback(Protocol):
    """定义Callback协议接口，由可调用对象实现。"""
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """用于异步回调的检查点处理程序，仅用于保存，不支持恢复。"""

    def __init__(self):
        self._executor = futures.ThreadPoolExecutor(max_workers=1)  # 初始化线程池执行者

    def close(self):
        self._executor.shutdown()  # 关闭线程池执行者

    def save(self, directory: epath.Path, args: "CallbackSave"):
        if jax.process_index() == 0:  # 确保只有主进程执行保存
            args.callback(directory)  # 执行用户提供的回调函数

    async def async_save(self, directory: epath.Path, args: "CallbackSave") -> list[futures.Future]:
        """;启动异步保存过程."""
        return [self._executor.submit(self.save, directory, args)]  # feature中的线程池需要使用submit方法来调用save。这个函数是异步的，返回一个future对象

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")  # 不支持恢复行为


@ocp.args.register_with_handler(CallbackHandler, for_save=True)# 注册回调处理程序
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback  # 保存的数据结构，其中包含回调函数


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...  # 存根类，仅供注册


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    """
    根据传入的训练状态分离出预测所需的参数。

    参数:
    state : TrainState - 当前训练状态。

    返回：
    tuple[TrainState, Params] - 返回新的训练状态和提取出的参数。
    """
    if state.ema_params is not None:  # 如果存在指数移动平均参数
        params = state.ema_params  # 使用EMA参数
        train_state = dataclasses.replace(state, ema_params=None)  # 替换状态，将EMA参数设置为None
    else:
        params = state.params  # 否则直接获取常规参数
        train_state = dataclasses.replace(state, params={})  # 清空状态中的参数

    return train_state, params  # 返回更新后的训练状态和参数


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    """
    将分离的参数重新合并到训练状态中，需要慎重考虑参数的存在性。

    参数:
    train_state : TrainState - 当前训练状态。
    params : dict[str, Params] - 新的参数字典。

    返回：
    TrainState - 合并后的训练状态。
    """
    if train_state.params:  # 如果原状态包含参数
        return dataclasses.replace(train_state, ema_params=params["params"])  # 更新EMA参数
    return dataclasses.replace(train_state, params=params["params"])  # 更新普通参数
