"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType  # 模型类型别名，引用_model中定义的模型类型
Filter: TypeAlias = nnx.filterlib.Filter  # 过滤器别名，引用nnx.filterlib中的Filter类

@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """决定资产的位置（例如，用于设置数据管道的标准统计）。

    这些资产将复制到检查点下的 `assets/asset_id` 目录。

    可以用于加载来自不同检查点的资产（例如基本模型检查点）或其他集中位置。例如，从基本模型检查点加载Trossen机器人规范统计，可以使用：

    ```
    AssetsConfig(
        assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # 资产目录。如果未提供，将使用配置文件中的assets_dirs。这对从不同的检查点（例如基本模型检查点）或其他集中位置加载资产很有用。
    assets_dir: str | None = None

    # 资产ID。如果未提供，将使用repo id。这允许用户引用描述不同机器人平台的资产。
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo ID。如果为None，则创建假数据。
    repo_id: str | None = None
    # 包含数据资产的资产目录内的子目录。
    asset_id: str | None = None
    # 包含预计算的归一化统计。如果为None，则不执行归一化。
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # 用于将输入从特定数据集格式转换为数据变换所期望的通用格式。dataclasses.field()函数用于定义默认值,默认值为一个空的_transforms.Group对象
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 数据变换，通常包括机器人特定的变换。在归一化之前应用。请参见`model.Observation`和`model.Actions`以了解关于归一化数据的信息。
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 模型特定的变换。在数据归一化后应用。
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 如果为true，将使用分位数归一化。否则，将使用正常的z-score归一化。
    use_quantile_norm: bool = False

    # 将用于数据加载器生成动作序列的键名。序列的长度由模型配置中的`action_horizon`字段定义。如果你的LeRobot数据集使用不同的键来表示动作，则应调整此项。
    action_sequence_keys: Sequence[str] = ("actions",)

    # 如果为true，将使用LeRobot数据集任务来定义提示。
    prompt_from_task: bool = False

    # 如果为true，将禁用从Hugging Face Hub同步数据集。允许在本地仅训练数据集。
    local_files_only: bool = False


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """创建一个组."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """为标准pi0模型创建模型变换。"""

    # 如果提供，将确定模型使用的默认提示。None表示没有默认提示，可以在构造的时候设置。
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:# 根据模型类型选择模型变换，类似cpp中的switch-case
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),  # 注入默认提示
                        _transforms.ResizeImages(224, 224),  # 调整图像大小
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),  # 使用PaligemmaTokenizer进行提示标记化
                        ),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),  # 对FAST输入进行标记化
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )  # 提取FAST动作
                    ],
                )

# 这里的abc.ABC是一个抽象基类，用于定义抽象方法，不能被实例化
@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # LeRobot repo ID。
    repo_id: str = tyro.MISSING
    # 决定如何加载资产。
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # 基础配置，将通过工厂更新。
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod# 抽象方法
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建数据配置。"""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None  # 获取repo_id，如果没有则为空
        asset_id = self.assets.asset_id or repo_id  # 若未提供asset_id则使用repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),  # 替换基础配置如果存在的话
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),  # 加载归一化统计
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None  # 无资产ID则返回None
        try:
            data_assets_dir = str(assets_dir / asset_id)  # 拼接资产目录
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))  # 加载归一化统计
            logging.info(f"Loaded norm stats from {data_assets_dir}")  # 日志记录
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")  # 处理找不到文件的异常
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    """创建一个简单的数据配置，用于测试。"""
    repo_id: str = "fake"  # 定义假数据配置的repo ID

    @override# 重写父类方法
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)  # 创建并返回简单的数据配置


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    """创建一个简单的数据配置，用于测试。"""
    # 数据变换的工厂。
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # 模型变换的工厂。
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs),  # 创建基础配置
            data_transforms=self.data_transforms(model_config),  # 应用数据变换
            model_transforms=self.model_transforms(model_config),  # 应用模型变换
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,  # 根据模型类型选择是否使用分位数归一化
        )

@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    """为LeRobot Aloha机器人创建数据配置。"""
    # 如果为真，将把关节维度根据当前状态转换为增量，传递给模型。夹具维度将保持绝对值。
    use_delta_joint_actions: bool = True
    # 如果提供，当“prompt”键不存在时将注入输入数据。
    default_prompt: str | None = None
    # 如果为真，这将把关节和夹具值从标准Aloha空间转换为base模型训练中使用的内部运行时的空间。使用标准Aloha数据的人应该将其设置为true。
    adapt_to_pi: bool = True

    # 重打包转化。
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},  # 映射输入图像
                        "state": "observation.state",  # 映射状态
                        "actions": "action",  # 映射动作为“action”
                    }
                )
            ]
        )
    )
    # 用于从数据集中读取动作序列的动作键。
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """
        创建数据配置的方法。
        
        :param assets_dirs: 配置资产的目录路径
        :param model_config: 基础模型配置
        :return: 返回数据配置对象
        """
        # 创建输入变换和输出变换
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)],  # 创建输入变换
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],  # 创建输出变换
        )
        
        # 当使用增量动作时创建一个掩码
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)  # 创建增量动作掩码
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],  # 推入增量变换
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],  # 推入绝对变换
            )

        # 创建模型变换，包括处理任何提示的信息
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)  # 创建模型变换

        return dataclasses.replace(
            self.create_base_config(assets_dirs),  # 创建基础配置
            repack_transforms=self.repack_transforms,  # 使用重打包转化
            data_transforms=data_transforms,  # 数据变换
            model_transforms=model_transforms,  # 模型变换
            action_sequence_keys=self.action_sequence_keys,  # 动作序列键
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """为LeRobot Libero机器人创建数据配置。"""
    
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """
        创建Libero机器人的数据配置方法。
        
        :param assets_dirs: 配置资产的目录路径
        :param model_config: 基础模型配置
        :return: 返回数据配置对象
        """
        # 将输入调整为来自Libero环境的格式
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # 为策略训练准备数据
        # 将图像转换为uint8 numpy数组，并添加掩码
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )
        # 使用增量动作（不适用于夹持器）
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # 模型变换包括标记提示和目标动作等操作
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """训练配置类，用于定义整个训练过程中的各种参数设置."""

    # 配置名称，必须唯一，用于引用此配置。
    name: tyro.conf.Suppress[str]
    # 项目名称。
    project_name: str = "openpi"
    # 实验名称将用于命名元数据和检查点目录。
    exp_name: str = tyro.MISSING

    # 定义模型配置，一些属性（action_dim、action_horizon 和 max_token_len）由所有模型共享
    # -- 请参见 BaseModelConfig。特定模型实现（例如：Pi0Config）继承自 BaseModelConfig 并可定义附加属性。
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # 权重加载器在模型初始化后可以选择性地从磁盘加载（可能部分）权重。
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # 指定哪些权重应该被冻结。
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # 决定要训练的数据。
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # 配置资产的基本目录（例如：归一化统计）。
    assets_base_dir: str = "./assets"
    # 检查点的基本目录。
    checkpoint_base_dir: str = "./checkpoints"

    # 在训练过程中会被随机生成器使用的随机种子。
    seed: int = 42
    # 全局批大小。
    batch_size: int = 32
    # 用于数据加载器的工作线程数。增加此数字会加速数据加载，但会增加内存和 CPU 的使用。
    num_workers: int = 2
    # 要运行的训练步骤（批次）的数量。
    num_train_steps: int = 30_000

    # 记录训练指标的频率（以步骤计）。
    log_interval: int = 100
    # 保存检查点的频率（以步骤计）。
    save_interval: int = 1000
    # 如果设置，则与 step % keep_period == 0 匹配的任何现有检查点都不会被删除。
    keep_period: int | None = 5000

    # 如果为真，在检查点目录已存在时将覆盖它。
    overwrite: bool = False
    # 如果为真，将从上一个检查点恢复训练。
    resume: bool = False

    # 如果为真，将启用 wandb 日志记录。
    wandb_enabled: bool = True

    # 用于向策略服务器传递元数据。
    policy_metadata: dict[str, Any] | None = None

    # 如果值大于1，则将启用FSDP并在人为指定的设备间分片；整体设备内存将减少，但训练可能会更慢。
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """获取此配置的资产目录。"""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """获取此配置的检查点目录。"""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """获取可训练参数的过滤器。"""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        """后处理初始化，确保起始条件有效性"""
        if self.resume and self.overwrite:
            raise ValueError("不能同时resume和overwrite。")

# 使用 `get_config` 如果你需要在代码中通过名称获取配置。
_CONFIGS = [
    #
    # 推理 Aloha 配置。
    #
    TrainConfig(
        name="pi0_aloha",  # 配置名称
        model=pi0.Pi0Config(),  # 模型配置
        data=LeRobotAlohaDataConfig(  # 数据配置
            assets=AssetsConfig(asset_id="trossen"),  # 资产配置，指定资产ID
        ),
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",  # 默认提示信息
        ),
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",  # 默认提示信息
        ),
    ),
    #
    # 推理 DROID 配置。
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0.Pi0Config(action_horizon=10),  # 设置动作视野为10
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),  # 资产配置，指定资产ID
            data_transforms=lambda model: _transforms.Group(  # 数据转换逻辑
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],  # 输入维度
                outputs=[droid_policy.DroidOutputs()],  # 输出设置
            ),
            base_config=DataConfig(
                prompt_from_task=True,  # 从任务生成提示
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),  # 快速DROID模型配置
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # 微调 Libero 配置。
    #
    TrainConfig(
        name="pi0_libero",
        model=pi0.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",  # 指定数据集的repo ID
            base_config=DataConfig(
                local_files_only=False,  # 设置为True以仅使用本地数据集
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),  # 权重加载器
        num_train_steps=30_000,  # 训练步数
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),  # 冻结过滤器
        ema_decay=None,  # 指数移动平均衰减
    ),
    TrainConfig(
        name="pi0_fast_libero",
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        model=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    #
    # 微调 Aloha 配置。
    #
    # 此配置用于演示如何在自定义 LeRobot 数据集上进行训练。
    # 有关如何转换和训练自己的 Aloha 数据集的说明，请参见 examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",  # 资产目录
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",  # 默认提示信息
            repack_transforms=_transforms.Group(  # 重打包变换
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # 设置为True以仅使用本地数据集
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,  # 训练步数
    ),
    # 此配置用于演示如何在简单模拟环境中进行训练。
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",  # 默认提示信息
            use_delta_joint_actions=False,  # 是否使用增量联合动作
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # 调试配置。
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),  # 假数据配置
        batch_size=2,  # 批大小
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),  # 调试用的模型配置
        save_interval=100,  # 保存间隔
        overwrite=True,  # 是否覆盖
        exp_name="debug",  # 实验名称
        num_train_steps=10,  # 训练步数
        wandb_enabled=False,  # 是否启用wandb
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),  # 加载检查点权重
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
]

# 检查配置名称是否唯一
if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")  # 抛出错误

# 将配置字典化，以便通过名称快速访问
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    """命令行接口，用于获取可覆盖的配置。"""
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """根据名称获取配置。"""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)  # 查找最接近的配置名
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""  # 提示用户可能的正确名称
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")  # 抛出错误

    return _CONFIGS_DICT[config_name]  # 返回对应的配置

