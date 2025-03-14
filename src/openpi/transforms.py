from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize



# https://zhuanlan.zhihu.com/p/585442275
DataDict: TypeAlias = at.PyTree  # 数据字典类型别名，为可能嵌套的字典结构。这里: TypeAlias 表示类型别名，at.PyTree 表示嵌套字典结构。具体使用时DataDict的类型为at.PyTree。
NormStats: TypeAlias = _normalize.NormStats  # 标准化统计信息类型别名。

T = TypeVar("T")  # 类型变量 T，用于泛型，表示任意类型，有点像cpp中的模板
S = TypeVar("S")  # 类型变量 S，用于泛型。

@runtime_checkable # 装饰器，这个类可以用于运行时类型检查
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """应用转换到数据上。

        参数:
            data: 要应用转换的数据。可能是一个嵌套字典，包含未批处理的数据元素。
                每个叶子节点被期望为 numpy 数组。允许使用 JAX 数组，但不推荐，因为这可能会导致在数据加载器工作进程中的额外 GPU 内存使用。

        返回:
            转换后的数据。可以是输入的 `data`（就地修改），也可以是一个新的数据结构。
        """

# 用于数据类的装饰器，frozen=True表示数据类是不可变的。就是在初始化对象后，不能再修改对象里的field
# 工厂方法来创建实例，这样可以在内部处理参数的赋值逻辑，
# 但是在实际使用的时候,也可以赋值（src/openpi/training/config.py）中
# 如果参数有默认值，那么在实例化对象时可以选择不传递这些参数
@dataclasses.dataclass(frozen=True)
class Group:
    """一组变换."""

    # 应用于模型输入数据的变换。这里使用：来表示input的类型为Sequence[DataTransformFn]，表示输入的变换函数序列，=()表示默认值为空序列，这个是类的属性
    inputs: Sequence[DataTransformFn] = ()

    # 应用于模型输出数据的变换。
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":#这里的-> "Group"表示返回值的类型为Group，*则是指定后面的参数只能通过关键字传递
        """将变换附加到当前组并返回新组。

        参数:
            inputs: 附加到当前输入变换的*末尾*。
            outputs: 附加到当前输出变换的*开头*.

        返回:
            一个新组，其中附加了变换。
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))#这里的*表示解包，将inputs和outputs解包后传入


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """组合变换，按顺序应用一系列变换."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:  # 遍历所有变换
            data = transform(data)  # 对数据进行变换
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """将一系列变换组合成单一变换."""
    return CompositeTransform(transforms)  # 返回复合变换对象


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """将输入字典重新打包为新字典.

    重新打包通过一个字典定义，新键是新键，值是旧键的扁平路径。使用 '/' 作为扁平化过程中的分隔符。

    示例:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]  # 结构描述的新旧键映射，这里[...]表示结构是一个列表，用来描述新旧键的映射关系

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)  # 将输入数据扁平化
        return jax.tree.map(lambda k: flat_item[k], self.structure)  # 根据新结构生成新字典


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None  # 默认提示内容

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:  # 如果没有提供提示，则添加
            data["prompt"] = np.asarray(self.prompt)  # 将提示转为数组格式
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    """对数据进行归一化处理.
    对每个叶子节点应用归一化操作。如果数据是嵌套字典，则根据归一化统计对每个叶子节点应用归一化操作。
    """
    norm_stats: at.PyTree[NormStats] | None  # 归一化统计，默认为 None
    use_quantiles: bool = False  # 是否使用分位数归一化
    strict: bool = False  # 严格模式下，是否检查所有统计关键字都在数据里

    # __post_init__在类的实例化后立即被调用的，用于初始化对象，传入的是上面的norm_stats、use_quantiles、strict三个参数
    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)  # 校验统计量完整性

    # __call__方法是DataTransformFn类的抽象方法，需要在子类中实现。实例化 Normalize 类的对象后，直接调用该对象时会触发
    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data  # 如果没有归一化统计则直接返回原始数据

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,  # 选择适当的归一化方法
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):  # Z-score 归一化实现
        return (x - stats.mean) / (stats.std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):  # 分位数归一化实现
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    """ 对数据进行反归一化处理.
    对每个叶子节点应用反归一化操作。如果数据是嵌套字典，则根据归一化统计对每个叶子节点应用反归一化操作。
    """
    norm_stats: at.PyTree[NormStats] | None  # 同 Normalize 类，反归一化统计
    use_quantiles: bool = False  # 同样控制是否使用分位数机制

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)  # 检查统计数据的有效性

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data  # 不需要反归一化时返回原数据

        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,  # 选择反向操作的方法
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):  # 垂直方向反归一化实现
        return x * (stats.std + 1e-6) + stats.mean

    def _unnormalize_quantile(self, x, stats: NormStats):  # 分位数反归一化实现
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    """调整图像大小.
    """
    height: int  # 图片目标高度
    width: int  # 图片目标宽度

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}  # 调整图片大小，从data中提取k,v，然后使用image_tools.resize_with_pad调整大小，并构成新的字典
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    """对动作进行下采样。"""
    stride: int  # 采样步长

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]  # 按照步长进行抽样
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """将绝对动作重包装为增量动作空间."""

    mask: Sequence[bool] | None  # 动作维度的布尔掩码

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data  # 没有动作数据或掩码则返回原数据

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)  # 根据状态计算增量动作
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """将增量动作重包装为绝对动作空间."""
    
    mask: Sequence[bool] | None  # 同相关属性

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data  # 无效情形返回原数据

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)  # 恢复至绝对动作
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    """对提示内容进行分词处理."""
    tokenizer: _tokenizer.PaligemmaTokenizer  # 自定义分词器

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")  # 提示内容必需

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt)  # 使用分词器进行分词
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}  # 返回更新的信息，其中**data表示原数据，"tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks表示新增的信息


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer  # FAST版分词器
    
    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")  # 提示内容必需

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)  # 快速分词处理
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    """从令牌中提取动作."""
    tokenizer: _tokenizer.FASTTokenizer 
    action_horizon: int  # 动作预测范围
    action_dim: int  # 动作维度 

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        tokens = data.pop("actions")  # 获取待处理的令牌
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)  # 从令牌提取动作
        return {
            **data,
            "actions": actions,  # 更新数据字典中的动作
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """从当前 LeRobot 数据集任务中提取提示."""

    tasks: dict[int, str]  # 任务索引与名称的对应关系

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')  # 必须提供任务索引

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")  # 未找到该任务的提示
        
        return {**data, "prompt": prompt}  # 返回更新的数据，包括所提取的提示


def flatten_dict(tree: at.PyTree) -> dict:
    """扁平化嵌套字典。使用 '/' 作为分隔符."""
    return traverse_util.flatten_dict(tree, sep="/")  # 调用工具类执行扁平化处理


def unflatten_dict(tree: dict) -> at.PyTree:
    """解扁平化字典。假设使用 '/' 作为分隔符."""
    return traverse_util.unflatten_dict(tree, sep="/")  # 拆解还原嵌套字典


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """根据给定模板转变嵌套字典的结构。

    转变以 `patterns` 字典定义。关键字是需要匹配的输入键，值是输出字典内的新名称。如果值为 None，则删除输入关键字。

    所有键和值均应表示使用 '/' 作为分隔符的扁平路径。
    键可以是正则表达式，值可包括对匹配组的反向引用（参见 `re.sub` 的更多细节）。请注意，正则表达式必须完全匹配整个关键字。

    在 `patterns` 字典内部的顺序很重要。仅使用第一个匹配输入键的模式。

    关于帮助例子的详细说明，请见单元测试。

    参数:
        patterns: 从旧键到新键的映射。
        tree: 被转换的嵌套字典。

    返回:
        转换后的嵌套字典。
    """
    data = flatten_dict(tree)  # 扁平化输入树

    # 编译模式
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():  # 浏览编译好的规则
            if pattern.fullmatch(k):  # 逐一比对
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None  # 替换关键字
                break
        else:
            new_k = k  # 若未匹配，保持原钥字

        if new_k is not None:
            if new_k in output:  # 防止产生重复键
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # 验证输出结构确保能够还原
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    # 将嵌套字典展平，以便于处理
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        # 如果键k在选择器中，则应用函数fn，将值v和相应的selector[k]作为参数传入
        if k in selector:
            return fn(v, selector[k])
        return v  # 否则返回原始值

    # 如果strict设置为True，检查所有选择器的键是否存在于树中
    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    # 返回应用完变换后的树结构，通过将转换后所有项组合成一个新字典
    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})

def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
    """将数组沿指定轴填充到目标维度，用零填补。"""
    current_dim = x.shape[axis]  # 获取当前维度大小
    if current_dim < target_dim:  # 如果当前维度小于目标维度
        pad_width = [(0, 0)] * len(x.shape)  # 初始化填充宽度列表
        pad_width[axis] = (0, target_dim - current_dim)  # 对所选轴进行填充
        return np.pad(x, pad_width)  # 使用np.pad执行填充并返回结果
    return x  # 如果当前维度已达目标维度，则返回原始数组

def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """为给定的维度生成布尔掩码。

    示例:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    参数：
        dims: 要创建掩码的维度。

    返回：
        布尔元组。
    """
    result = []  # 初始化结果列表
    for dim in dims:
        if dim > 0:  # 如果维度大于0，添加对应的True
            result.extend([True] * (dim))
        else:  # 如果维度小于等于0，添加对应的False
            result.extend([False] * (-dim))
    return tuple(result)  # 返回由布尔值组成的元组

def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    # 遍历归一化统计数据，如果缺少q01或q99，则抛出异常
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
